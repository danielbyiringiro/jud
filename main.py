import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
from scipy.sparse import load_npz
import pickle

# Load data for new.py
@st.cache_data
def load_new_data():
    merged_data = pd.read_csv("data.csv")
    products = pd.read_csv("products.csv")
    customers = pd.read_csv("customers.csv")
    return merged_data, products, customers

# Load data for main.py
@st.cache_data
def load_main_data():
    df = pd.read_csv('product_data.csv')
    df_image = pd.read_csv('original.csv')
    tfidf_matrix = load_npz('tfidf_matrix.npz')
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return df, df_image, tfidf_matrix, vectorizer

# Function to recommend products based on user's previous purchases (from new.py)
def recommend_products(user_id, num_recommendations=5):
    merged_data, products, _ = load_new_data()
    user_purchases = merged_data[merged_data['customer_id'] == user_id]
    
    if user_purchases.empty:
        return "No purchases found for this user."
    
    purchased_products = user_purchases['title'].unique()
    
    if not purchased_products.size:
        return "No products found for recommendation."
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(products['title'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    recommended_products = set()
    for product in purchased_products:
        idx = products[products['title'] == product].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]
        product_indices = [i[0] for i in sim_scores]
        recommended_products.update(products[['product_id', 'title']].iloc[product_indices]['product_id'])
    
    recommended_products.difference_update(user_purchases['product_id'])
    user_purchases_id = user_purchases['product_id']
    
    if not recommended_products:
        return "No new products to recommend."
    
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

        st.write("---")

# Main Streamlit app
df, df_image, tfidf_matrix, vectorizer = load_main_data()

# Tabbed interface
tab1, tab2 = st.tabs(["Fashion Recommendation", "Product Recommendation System"])

with tab1:
    st.title("Fashion Recommendation")
    _, _, customers = load_new_data()
    selected_name = st.selectbox('Choose a name:', customers['name'])
    if selected_name:
        id = customers[customers.name == selected_name]['id'].item()
        rec, purchases = get_recco(int(id))
        st.subheader("Purchased Products")
        show_results(purchases)
        st.subheader("Recommended Products")
        show_results(rec)

with tab2:
    st.title("Product Recommendation System")
    query = st.text_input("Enter your query (e.g., 'I am looking for a plain tshirt'):")

    if query:
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-5:][::-1]
        st.write("Top 5 Recommended Products:")
        
        for idx in top_indices:
            product_id = df.iloc[idx]['product_id']
            image_path_1 = df_image[df_image.product_id == product_id]['image_1'].item()
            image_path_2 = df_image[df_image.product_id == product_id]['image_2'].item()
            path_1 = f"products/{image_path_1}"
            path_2 = f"products/{image_path_2}"
            
            image_1 = Image.open(path_1)
            image_2 = Image.open(path_2)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image_1, caption="Product Image", use_column_width=True)
                st.image(image_2, caption="Product Sub Image", use_column_width=True)

            with col2:
                st.write(f"**Title:** {df.iloc[idx]['title']}")
                st.write(f"**Category:** {df.iloc[idx]['category']}")
                st.write(f"**Price:** GHc {df.iloc[idx]['price']}")
                st.write(f"**Description:** {df.iloc[idx]['descriptions']}")
                st.write(f"**Color:** {df.iloc[idx]['color']}")
                st.write(f"**Size:** {df.iloc[idx]['size']}")
                st.write(f"**Weight:** {df.iloc[idx]['weight']} kg")

            st.write("---")
