import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image

st.title("Fashion Recommendation")

@st.cache_data
def load_data():
    merged_data = pd.read_csv("data.csv")
    products = pd.read_csv("products.csv")
    customers = pd.read_csv("customers.csv")
    return merged_data, products, customers

# Function to recommend products based on user's previous purchases
def recommend_products(user_id, num_recommendations=5):
    # Get products bought by the user
    merged_data, products, _ = load_data()
    user_purchases = merged_data[merged_data['customer_id'] == user_id]
    
    if user_purchases.empty:
        return "No purchases found for this user."
    
    # Get the list of unique products the user has bought
    purchased_products = user_purchases['title'].unique()
    
    if not purchased_products.size:
        return "No products found for recommendation."
    
    # Create TF-IDF features for all products
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(products['title'])
    
    # Calculate cosine similarity between products
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get recommendations based on purchased products
    recommended_products = set()
    for product in purchased_products:
        idx = products[products['title'] == product].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]  # Exclude the product itself
        product_indices = [i[0] for i in sim_scores]
        recommended_products.update(products[['product_id', 'title']].iloc[product_indices]['product_id'])
    
    # Remove products already bought by the user
    recommended_products.difference_update(user_purchases['product_id'])
    user_purchases_id = user_purchases['product_id']
    
    if not recommended_products:
        return "No new products to recommend."
    
    # Fetch recommended products details
    recommended_products_details = products[products['product_id'].isin(recommended_products)]
    user_purchases = products[products['product_id'].isin(user_purchases_id)]
    
    return recommended_products_details, user_purchases

def get_recco(user_id):
    rec, purchases = recommend_products(user_id)
    final = rec[rec.available == "Yes"]
    return final, purchases

def show_results(purchases):
    for _, row in purchases.iterrows():
        image_1 = row['image_1']
        path_1 = f"/opt/lampp/htdocs/afrikana/assets/images/products/{image_1}"
        image_1 = Image.open(path_1)

        col1, col2 = st.columns([1,2])
        with col1:
            st.image(image_1, caption="Product Image", use_column_width=True)
        with col2:
            st.write(f"**Title:** {row['title']}")
            st.write(f"**Category:** {row['category']}")
            st.write(f"**Price:** GHc {row['price']}")
            st.write(f"**Description:** {row['descriptions']}")
            st.write(f"**Color:** {row['color']}")
            st.write(f"**Size:** {row['size']}")
            st.write(f"**Weight:** {row['weight']} kg")

        # Divider
        st.write("---")


_, _, customers = load_data()
selected_name = st.selectbox('Choose a name:', customers['name'])
if selected_name:
    id = customers[customers.name == selected_name]['id'].item()
    rec, purchases = get_recco(int(id))
    st.subheader("Purchased Products")
    show_results(purchases)
    st.subheader("Recommended Products")
    show_results(rec)