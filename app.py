import ast
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model
from flask import Flask, request, jsonify, render_template
import os
import pickle
from munkres import Munkres

app = Flask(__name__, static_folder='static', template_folder='templates')

# تنظیم مسیرهای دایرکتوری
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # مسیر اصلی پروژه
MODEL_DIR = os.path.join(BASE_DIR, "models")  # پوشه مدل‌ها
DATA_DIR = os.path.join(BASE_DIR, "data")  # پوشه داده‌ها
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")  # پوشه ذخیره پیش‌بینی‌ها

# ایجاد پوشه‌ها در صورت عدم وجود
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# بارگذاری مدل LSTM
model_filename = os.path.join(MODEL_DIR, "lstm_model.h5")
if os.path.exists(model_filename):
    combined_model100_3 = load_model(model_filename)
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found: {model_filename}")

# بارگذاری داده‌های ترکیبی
data_filename = os.path.join(DATA_DIR, "combined_dataset_dgi100.pkl")
if os.path.exists(data_filename):
    with open(data_filename, 'rb') as file:
        combined_dataset_dgi100 = pickle.load(file)
    print("combined_dataset_dgi100 loaded successfully!")
else:
    raise FileNotFoundError(f"Data file not found: {data_filename}")


# مسیر ذخیره مدل
file_path = "D:\\bagheri"    
model_filename = "lstm_model2.h5"
model_path = os.path.join(file_path, model_filename)

# بارگذاری مدل
model_lstm = load_model(model_path)


# صفحه اصلی
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/plot')
def show_plot():
    return render_template("plot.html")

@app.route('/predict_embeddings', methods=['POST'])
def predict_embeddings():
    try:
        data = request.json  # دریافت داده‌های ورودی از JSON
        year = int(data.get("year", None))

        if year is None or not (1988 <= year <= 2024):
            return jsonify({"error": "سال نامعتبر است. لطفا یک سال بین 1988 تا 2024 انتخاب کنید"}), 400

        # استخراج داده‌های مورد نیاز برای پیش‌بینی
        combined_input_sequences_node_test_dgi100 = []
        combined_test_words_and_start_year_dgi100 = []
        combined_output_sequences_node_test_dgi100 = []

        for sample in combined_dataset_dgi100:
            start_year = sample['word and start year'][1]

            if start_year == year - 3:
                embeddings = sample['embeddings']
                if len(embeddings) < 4:
                    continue
                
                combined_input_sequences_node_test_dgi100.append(embeddings[:3])
                combined_output_sequences_node_test_dgi100.append(embeddings[3])
                combined_test_words_and_start_year_dgi100.append(sample['word and start year'])

        if not combined_input_sequences_node_test_dgi100:
            return jsonify({"message": "سال وارد شده خارج از محدوده داده‌ها است."}), 404
        else:
            # تبدیل لیست به آرایه Numpy
            combined_input_sequences_node_test_dgi100 = np.array(combined_input_sequences_node_test_dgi100)
            combined_input_sequences_node_test_dgi100 = combined_input_sequences_node_test_dgi100.reshape(
                combined_input_sequences_node_test_dgi100.shape[0], 
                combined_input_sequences_node_test_dgi100.shape[1], 
                combined_input_sequences_node_test_dgi100.shape[2]
                )

        # انجام پیش‌بینی
        predictions1 = combined_model100_3.predict(combined_input_sequences_node_test_dgi100)

        # ذخیره پیش‌بینی‌ها
        predictions_file = os.path.join(PREDICTIONS_DIR, f"predictions_{year}.pkl")
        with open(predictions_file, "wb") as file:
            pickle.dump(predictions1, file)

        # Persian message to return
        print("word2vec prediction made successfully")
        # اضافه کردن کد برای محاسبه ارتباط بین خوشه‌ها
        # بارگذاری داده‌های نرمال‌شده
        file_path = "D:\\bagheri"
        normalized_df = pd.read_csv(f'{file_path}/normalized_windowed_cluster_relationships.csv')

    
        testing_data_lstm = normalized_df[normalized_df['start_year'] == year - 3]
        testing_data_lstm = testing_data_lstm.reset_index(drop=True)
        testing_data_lstm['normalized_weights'] = testing_data_lstm['normalized_weights'].apply(ast.literal_eval)
        

        # آماده‌سازی داده‌های تست برای LSTM
        X_test_lstm = np.array([sample[:-1] for sample in testing_data_lstm['normalized_weights']])
        X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

        # پیش‌بینی با مدل LSTM
        predictions_lstm = model_lstm.predict(X_test_lstm)
        print("predictions_lstm made successfully")

        # درنظر گرفتن وزن‌های پیش‌بینی‌شده
        predicted_weights_lstm = predictions_lstm
        test_cluster_pairs_lstm = testing_data_lstm['cluster_pair']

        # دیکشنری برای نگهداری وزن‌های پیش‌بینی‌شده
        predicted_cluster_weights_lstm = {}

        # نگاشت وزن‌های پیش‌بینی‌شده به جفت خوشه‌ها
        for cluster_pair, weight in zip(test_cluster_pairs_lstm, predicted_weights_lstm):
            predicted_cluster_weights_lstm[cluster_pair] = weight

        # معکوس کردن نرمال‌سازی وزن‌ها
        original_weights = []
        for i, (cluster_pair, predicted_weight) in enumerate(zip(test_cluster_pairs_lstm, predictions_lstm)):
            # تبدیل normalized_weights_3_years به نوع عددی
            normalized_weights_3_years = np.array(ast.literal_eval(testing_data_lstm[testing_data_lstm['cluster_pair'] == cluster_pair]['weights'].values[0]))
    
            # تبدیل predicted_weight به نوع float
            predicted_weight = float(predicted_weight)
    
            # ترکیب وزن‌ها
            all_weights = np.append(normalized_weights_3_years, predicted_weight)
    
            # محاسبه میانگین و انحراف معیار
            mean_weight = np.mean(all_weights)
            std_weight = np.std(all_weights)
    
            # معکوس کردن نرمال‌سازی
            original_weight = predicted_weight * std_weight + mean_weight
            original_weights.append((cluster_pair, original_weight))


        # دیکشنری از وزن‌های اصلی
        predicted_original_weights_lstm = dict(original_weights)

        # مرتب‌سازی بر اساس وزن‌ها و گرفتن 5 جفت خوشه با بالاترین وزن
        sorted_weights_lstm = sorted(predicted_original_weights_lstm.items(), key=lambda x: x[1], reverse=True)
        top_5_cluster_pairs_lstm = sorted_weights_lstm[:5]

        # Perform clustering for the year 2024
        # Use predicted embeddings and words
        # Create separate lists for predicted embeddings
        predicted_embeddings = []

        # Iterate through predictions and extract the words and embeddings
        for prediction in predictions1:
            predicted_embeddings.append(prediction)
        # Create separate lists for words and their corresponding predicted embeddings
        wordsyear = []
        actual_embeddings_year = []

        # Iterate through predictions and extract the words and embeddings
        for i, embedding in enumerate(combined_output_sequences_node_test_dgi100):
            word, start_year = combined_test_words_and_start_year_dgi100[i]
            wordsyear.append(word)
            actual_embeddings_year.append(embedding)


        kmeans = KMeans(n_clusters=10, random_state=42)
        predicted_cluster_labels_year = kmeans.fit_predict(predicted_embeddings)

        # predicted centroids for 2024
        predicted_centroids_year = kmeans.cluster_centers_

        # Create a dictionary to store words in each predicted cluster for 2024
        predicted_clusters_year = {i: [] for i in range(10)}  # Initialize 10 empty clusters
        for word, cluster_label in zip(wordsyear, predicted_cluster_labels_year):
            predicted_clusters_year[cluster_label].append(word)

        # Save yearly clusters and centroids to pickle files
        file_path = "D:\\bagheri"
        with open(f'{file_path}/yearly_clusters.pkl', 'rb') as clusters_file:
            yearly_clusters = pickle.load(clusters_file)

        with open(f'{file_path}/yearly_centroids.pkl', 'rb') as centroids_file:
            yearly_centroids = pickle.load(centroids_file)
        
        with open(f'{file_path}/consistent_cluster_names.pkl', 'rb') as f:
            cluster_names = pickle.load(f)

        # Initialize Munkres instance
        munkres = Munkres()

        current_centroids = predicted_centroids_year
        previous_centroids = yearly_centroids[(year - 1)]

        current_clusters = predicted_clusters_year
        previous_clusters = yearly_clusters[(year - 1)]

        # Compute cosine similarity matrix (centroid similarity)
        centroid_similarity = cosine_similarity(current_centroids, previous_centroids)

        # Compute Jaccard similarity matrix (word overlap)
        word_overlap_similarity = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                words_1 = set(current_clusters[i])
                words_2 = set(previous_clusters[j])
                if len(words_1 | words_2) > 0:  # Avoid division by zero
                    word_overlap_similarity[i, j] = len(words_1 & words_2) / len(words_1 | words_2)

        # Combine similarities (weighted sum)
        combined_similarity = 0.7 * centroid_similarity + 0.3 * word_overlap_similarity

        # Convert similarity to a cost matrix (negate values for Munkres)
        cost_matrix = -combined_similarity

        # Compute optimal mapping using Munkres
        indices = munkres.compute(cost_matrix)  # List of (current_cluster, previous_cluster) pairs

        # Map clusters and save the mapping
        cluster_mapping = {}
        for current_cluster, previous_cluster in indices:
            cluster_mapping[current_cluster] = previous_cluster
        predicted_cluster_mapping_year = cluster_mapping

        # Assign consistent cluster names based on the mapping
        predicted_cluster_names_year = {}
        for current_cluster, previous_cluster in cluster_mapping.items():
            predicted_cluster_names_year[current_cluster] = cluster_names[(year - 1)][previous_cluster]

        # Get clusters and cluster names for actual 2024
        clusters = predicted_clusters_year
        cluster_name_mapping = predicted_cluster_names_year  # Consistent cluster names for the year

        # Rename clusters based on consistent names
        renamed_clusters = {}
        for cluster_id, words in clusters.items():
            renamed_cluster_name = cluster_name_mapping[cluster_id]
            renamed_clusters[renamed_cluster_name] = words

        # Save renamed clusters for the year
        predicted_updated_yearly_clusters_year = renamed_clusters

        # # --- Example: Access updated clusters ---
        # print(f"Updated predicted clusters for year:", predicted_updated_yearly_clusters_year)
        # انتخاب جفت خوشه دارای بالاترین وزن
        highest_weight_cluster_pair_str, highest_weight = top_5_cluster_pairs_lstm[0]  # مثال: ('Cluster_3', 'Cluster_7')
        highest_weight_cluster_pair = ast.literal_eval(highest_weight_cluster_pair_str)  
        

        # دریافت کلمات داخل هر خوشه
        cluster_1, cluster_2 = highest_weight_cluster_pair
        print('cluster_1: ', cluster_1)
        print('cluster_2: ', cluster_2)
        print('Available clusters:', predicted_updated_yearly_clusters_year.keys())
        words_in_cluster_1 = predicted_updated_yearly_clusters_year.get(cluster_1, [])
        print('words_in_cluster_1: ', words_in_cluster_1)
        words_in_cluster_2 = predicted_updated_yearly_clusters_year.get(cluster_2, [])
        print('words_in_cluster_2: ', words_in_cluster_2)
        print("Type of words_in_cluster_1:", type(words_in_cluster_1))
        print("Type of words_in_cluster_2:", type(words_in_cluster_2))
        try:
            json.dumps({
                "words_in_cluster_1": words_in_cluster_1,
                "words_in_cluster_2": words_in_cluster_2
            })
            print("JSON test passed ✅")
        except Exception as e:
            print("JSON test failed ❌:", e)



        # نمایش 5 جفت خوشه برتر و وزن‌های آن‌ها
        top_5_message = ""
        for (cluster_pair, weight) in top_5_cluster_pairs_lstm:
            top_5_message += f"Cluster Pair: {cluster_pair}, Weight: {weight}\n"

        return jsonify({
            "top_5_cluster_pairs": [f"Cluster Pair: {cluster_pair}, Weight: {weight}" for cluster_pair, weight in top_5_cluster_pairs_lstm],
            "highest_weight_cluster": {
                "cluster_pair": highest_weight_cluster_pair,
                "weight": highest_weight,
                "cluster_1_name": cluster_1,  # اضافه کردن نام خوشه اول
                "cluster_2_name": cluster_2, 
                "words_in_cluster_1": words_in_cluster_1,
                "words_in_cluster_2": words_in_cluster_2
                }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
