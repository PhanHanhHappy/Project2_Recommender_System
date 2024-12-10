import streamlit as st
import pandas as pd
import pickle
from surprise import Reader, Dataset, SVD

###### Giao diện Streamlit ######
st.image('01_hasakilogo.jpg', use_container_width=True)

st.title("Đồ án tốt nghiệp Data Science")
st.header("Topic: Recommender System")

# Chia layout thành 3 cột
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    st.markdown(
        """
        <div style="padding: 10px; border-radius: 5px;">
            <h5>Họ và tên nhóm 4:</h5>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.image('15_memberpicture.jpg', use_container_width=True)
with st.container():
    st.markdown(
        """
        <style>
        .stContainer {
            border: 1px solid #78b379;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    font_size = 24
    st.write(f"<p style='font-size: {font_size}px;'>Giáo viên hướng dẫn: Khuất Thùy Phương</p>", unsafe_allow_html=True)
    font_size = 24
    st.write(f"<p style='font-size: {font_size}px;'>Ngày báo cáo: 14/12/2024</p>", unsafe_allow_html=True)

# Tiêu đề trang
st.title("Menu")

# Tạo menu sổ xuống (Dropdown)
menu_options = ['Mục tiêu - Nội dung', 'Cosine', 'Surprise', 'Hybrid']
# Tạo st.selectbox
with st.container():
    st.markdown(
        "<p style='font-size: 20px; font-weight: bold; margin-bottom: 5px;'>Chọn một mục:</p>",  # Giảm margin-bottom
        unsafe_allow_html=True
    )
    selected_option = st.selectbox('--', menu_options)

# Đọc dữ liệu
df_products = pd.read_csv('GUIthird_san_pham_processed_3nd.csv')
danh_gia = pd.read_csv('Danh_gia_filtered.csv')
khach_hang = pd.read_csv('Khach_hang.csv')
danh_gia2=pd.read_csv("Danh_gia.csv")

# Tải mô hình content-based
with open('products_cosine_sim.pkl', 'rb') as f:
    cosine_sim_new = pickle.load(f)

 # Tải mô hình SVD đã lưu
with open('svd_model.pkl', 'rb') as f:
    model_svd = pickle.load(f)

# --- Các hàm cho phần Cosine ---
def get_recommendations(df, ma_san_pham, cosine_sim, nums=3):
    # Get the index of the product that matches the ma_san_pham
    matching_indices = df.index[df['ma_san_pham'] == ma_san_pham].tolist()
    if not matching_indices:
        print(f"No product found with ID: {ma_san_pham}")
        return pd.DataFrame()  # Return an empty DataFrame if no match
    idx = matching_indices[0]

    # Get the pairwise similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the nums most similar products (Ignoring the product itself)
    sim_scores = sim_scores[1:nums+1]

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Lấy các hàng theo product_indices
    df_filtered = df.iloc[product_indices]

    # Lọc sản phẩm có điểm trung bình lớn hơn 4
    df_filtered = df_filtered[df_filtered['diem_trung_binh'] > 4]

    return df_filtered

def display_recommended_products(recommended_products, cols=3):
    for i in range(0, len(recommended_products), cols):
        cols = st.columns(cols)
        for j, col in enumerate(cols):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                with col:   
                    st.write(product['ten_san_pham'])
                    st.write(f"Giá bán: {product['gia_ban']}")
                    st.write(f"Giảm giá: {product['ty_le_giam_gia']}")
                    st.write(f"Điểm trung bình: {product['diem_trung_binh']}")
                    st.write(f"Volume: {product['volume']}")
                    
                    expander = st.expander(f"Mô tả")
                    product_description = product['mo_ta']
                    truncated_description = ' '.join(product_description.split()[:100]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")           

#--- Các hàm cho surprise---
# - Hàm kiểm tra user_id là mới hay cũ
def is_new_user(user_id):
    """
    Kiểm tra người dùng mới hay cũ dựa trên các điều kiện từ file 'danh_gia2' và 'khach_hang'.
    """
    # Kiểm tra xem 'danh_gia2' và 'khach_hang' đã được load vào session_state chưa
    if 'danh_gia2' not in st.session_state:
        st.session_state.danh_gia2 = pd.read_csv('Danh_gia_filtered.csv')
    if 'khach_hang' not in st.session_state:
        st.session_state.khach_hang = pd.read_csv('Khach_hang.csv')
    # Điều kiện xác định người dùng mới hoặc cũ
    if not st.session_state.khach_hang[st.session_state.khach_hang['ma_khach_hang'] == user_id].empty and \
       not st.session_state.danh_gia2[st.session_state.danh_gia2['ma_khach_hang'] == user_id].empty:
        return False  # User cũ
    else:
        return True  # User mới
    # Lấy DataFrame từ session_state
    danh_gia2 = st.session_state.danh_gia2
    khach_hang = st.session_state.khach_hang

def recommend_products_for_old_user(user_id, model, danh_gia2, df_products, top_n=3):
    """
    Hàm gợi ý sản phẩm cho người dùng cũ dựa trên mô hình SVD và điểm số ước tính.
    """
    item_ids = df_products['ma_san_pham'].unique()  # Lấy danh sách các ID sản phẩm
    products_and_scores = []

    # Dự đoán điểm số cho từng sản phẩm
    for item_id in item_ids:
        prediction = model.predict(user_id, item_id)  # Dự đoán điểm số cho từng sản phẩm
        products_and_scores.append((item_id, prediction.est))  # Lưu lại ID sản phẩm và điểm số dự đoán

    # Sắp xếp sản phẩm theo điểm số dự đoán giảm dần
    products_and_scores.sort(key=lambda x: x[1], reverse=True)

    # Lấy top_n sản phẩm có điểm số cao nhất
    top_products = [item_id for item_id, score in products_and_scores[:top_n]]

    # Lọc và trả về DataFrame với các cột mong muốn và điểm trung bình >= 4
    recommendations = df_products[df_products['ma_san_pham'].isin(top_products)]
    recommendations = recommendations[recommendations['diem_trung_binh'] >= 4]

    # Thêm cột 'estimate_score' vào DataFrame kết quả
    recommendations['estimate_score'] = [score for item_id, score in products_and_scores[:top_n]]

    # Tính tỷ lệ giảm giá và làm tròn
    recommendations['ty_le_giam_gia'] = ((recommendations['gia_goc'] - recommendations['gia_ban']) / recommendations['gia_goc'] * 100).round(0)

    return recommendations[['ma_san_pham', 'ten_san_pham', 'gia_ban', 'ty_le_giam_gia', 'diem_trung_binh', 'estimate_score', 'mo_ta', 'volume']]

# --- Hàm gợi ý sản phẩm cho người dùng mới kết hợp với SVD ---
def recommend_products_for_new_user_with_svd(user_id, model_svd, df_products, top_n=10):
    """
    Hàm gợi ý sản phẩm cho người dùng mới dựa trên mô hình SVD kết hợp với các yếu tố như điểm trung bình,
    số lượng đánh giá và tỷ lệ giảm giá.
    """
    # Tính điểm trung bình và số lượng đánh giá
    df_avg_ratings = df_products.groupby('ma_san_pham')['diem_trung_binh'].agg(['mean', 'count']).reset_index()
    df_avg_ratings.columns = ['ma_san_pham', 'avg_rating', 'rating_count']

    # Tính toán tỷ lệ giảm giá
    df_products['ty_le_giam_gia'] = (df_products['gia_goc'] - df_products['gia_ban']) / df_products['gia_goc']

    # Dự đoán điểm số từ mô hình SVD cho từng sản phẩm
    item_ids = df_products['ma_san_pham'].unique()
    product_scores = []
    for item_id in item_ids:
        prediction = model_svd.predict(user_id, item_id)
        product_scores.append((item_id, prediction.est))

    # Nối dữ liệu
    df_merged = pd.merge(df_products, df_avg_ratings, on='ma_san_pham')

    # Tính điểm kết hợp
    w_rating = 0.3
    w_count = 0.3
    w_discount = 0.2
    w_svd = 0.2
    df_merged['combined_score'] = (
        w_rating * df_merged['avg_rating'] +
        w_count * df_merged['rating_count'] +
        w_discount * df_merged['ty_le_giam_gia'] +
        w_svd * df_merged['ma_san_pham'].apply(lambda x: next((score for item_id, score in product_scores if item_id == x), 0))
    )

    # Lọc sản phẩm có điểm trung bình >= 4
    df_merged = df_merged[df_merged['diem_trung_binh'] >= 4]

    # Lấy top_n sản phẩm có điểm kết hợp cao nhất
    recommendations = df_merged.nlargest(top_n, 'combined_score')

    # Thêm cột phần trăm giảm giá và làm tròn đến 2 chữ số thập phân
    recommendations['ty_le_giam_gia'] = ((recommendations['gia_goc'] - recommendations['gia_ban']) / recommendations['gia_goc'] * 100).round(0)

    # Thêm cột 'combined_score' vào DataFrame kết quả
    recommendations['combined_score'] = recommendations['combined_score'].round(2)

    return recommendations[['ma_san_pham', 'ten_san_pham', 'gia_ban', 'ty_le_giam_gia', 'diem_trung_binh', 'combined_score', 'mo_ta', 'volume']]

#-----hàm cho kết hợp content-based và colabrative filtering-----
# - Hàm kiểm tra user_id là mới hay cũ (đã đổi tên thành check_if_new_user)
# Đọc dữ liệu khi khởi động ứng dụng
if 'danh_gia2' not in st.session_state:
    st.session_state.danh_gia2 = pd.read_csv('Danh_gia_filtered.csv')
if 'khach_hang' not in st.session_state:
    st.session_state.khach_hang = pd.read_csv('Khach_hang.csv')
if 'df_products' not in st.session_state:
    st.session_state.df_products = pd.read_csv('GUIthird_san_pham_processed_3nd.csv') 

def check_if_new_user(user_id, danh_gia, khach_hang):
    """
    Kiểm tra người dùng mới hay cũ dựa trên các điều kiện từ file 'danh_gia' và 'khach_hang'.
    """
    # Kiểm tra xem user_id có trong bảng 'danh_gia' không (đã có đánh giá)
    user_ratings = danh_gia[danh_gia['ma_khach_hang'] == user_id]

    # Kiểm tra trong bảng 'khach_hang' xem người dùng đã đăng ký chưa
    user_info = khach_hang[khach_hang['ma_khach_hang'] == user_id]

    if not user_info.empty:  # Nếu có thông tin trong 'khach_hang', là user cũ
        return False  # User cũ
    elif not user_ratings.empty:  # Nếu có đánh giá trong 'danh_gia', là user cũ
        return False  # User cũ
    else:
        return True  # Nếu không có đánh giá và không có thông tin trong 'khach_hang', là user mới

def simple_hybrid_recommendation(user_id, danh_gia, df_products, cosine_sim_new, model_svd, top_n=10, weight_content=0.5):
    """
    Kết hợp đơn giản hai mô hình gợi ý (Content-based và Collaborative Filtering) 
    cho người dùng, đồng thời xử lý trường hợp người dùng mới hoặc cũ.
    """

    # Gọi hàm check_if_new_user với đầy đủ tham số
    is_new_user = check_if_new_user(user_id, danh_gia, st.session_state.khach_hang)  
    # Tạo dictionary lưu trữ thông tin sản phẩm
    product_info = df_products.set_index('ma_san_pham').to_dict('index')

    # Khởi tạo recommendations_df (để tránh UnboundLocalError)
    recommendations_df = pd.DataFrame()

    if is_new_user:
        print("Đây là người dùng mới.")

        # 1. Chiến lược gợi ý cho người dùng mới (Sản phẩm phổ biến hoặc có điểm cao)
        high_rated_items = df_products.groupby('ma_san_pham')['diem_trung_binh'].mean()
        high_rated_items = high_rated_items[high_rated_items > 4].index.tolist()
        popular_items = danh_gia['ma_san_pham'].value_counts().head(top_n).index.tolist()
        recommendations = list(set(high_rated_items + popular_items))[:top_n]

        # 2. Tính toán Cosine Similarity cho các sản phẩm phổ biến và có điểm cao
        content_recommendations = []
        for item_id in recommendations:
            try:
                idx = df_products[df_products['ma_san_pham'] == item_id].index[0]
                sim_scores = list(enumerate(cosine_sim_new[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:top_n+1]
                content_recommendations.extend([df_products['ma_san_pham'].iloc[i[0]] for i in sim_scores])
            except IndexError:
                print(f"Lỗi IndexError với item_id: {item_id}")
                continue

        # 3. Tính toán SVD cho người dùng mới
        all_items = df_products['ma_san_pham'].tolist()
        predictions = [model_svd.predict(user_id, item_id) for item_id in all_items]
        svd_scores = {str(pred.iid): pred.est for pred in predictions}
        cf_recommendations = sorted(svd_scores, key=svd_scores.get, reverse=True)[:top_n]

        # Kết hợp Cosine Similarity và SVD
        combined_recommendations = list(set(content_recommendations + cf_recommendations))

        # 4. Xếp hạng lại dựa trên điểm số từ cả hai mô hình
        hybrid_scores = {}
        content_scores_result = {}
        svd_scores_result = {}
        for item_id in combined_recommendations:
            content_score = 0
            if item_id in content_recommendations:
                content_score = 1 / (content_recommendations.index(item_id) + 1)
            cf_score = svd_scores.get(str(item_id), 0)
            hybrid_scores[item_id] = weight_content * content_score + (1 - weight_content) * cf_score
            content_scores_result[item_id] = content_score
            svd_scores_result[item_id] = cf_score

        # 5. Tạo DataFrame kết quả (đã thêm các cột từ df_products)
        recommendations_df = pd.DataFrame({
            'ma_san_pham': list(hybrid_scores.keys()),
            'ten_san_pham': [product_info.get(int(item_id), {}).get('ten_san_pham') for item_id in hybrid_scores.keys()],
            'hybrid_score': list(hybrid_scores.values()),
            'content_score': list(content_scores_result.values()),
            'svd_score': list(svd_scores_result.values()),
            'diem_trung_binh': [product_info.get(int(item_id), {}).get('diem_trung_binh') for item_id in hybrid_scores.keys()],
            'gia_ban': [product_info.get(int(item_id), {}).get('gia_ban') for item_id in hybrid_scores.keys()],
            'ty_le_giam_gia': [product_info.get(int(item_id), {}).get('ty_le_giam_gia') for item_id in hybrid_scores.keys()],
            'mo_ta': [product_info.get(int(item_id), {}).get('mo_ta') for item_id in hybrid_scores.keys()],
            'volume': [product_info.get(int(item_id), {}).get('volume') for item_id in hybrid_scores.keys()]
        })

    else:
        print("Đây là người dùng cũ.")

        # Lấy các sản phẩm đã được người dùng đánh giá
        rated_items = danh_gia[danh_gia['ma_khach_hang'] == user_id]['ma_san_pham'].tolist()
        all_items = df_products['ma_san_pham'].tolist()
        all_item_names = df_products['ten_san_pham'].tolist()
        all_item_ratings = df_products['diem_trung_binh'].tolist()

        # 1. Lọc các sản phẩm có điểm trung bình > 4
        high_rated_items = [item for item in all_items if df_products.loc[df_products['ma_san_pham'] == item, 'diem_trung_binh'].values[0] > 4]

        # 2. Tạo danh sách gợi ý từ content-based
        content_recommendations = []
        for item_id in rated_items:
            if item_id in high_rated_items:
                idx = df_products[df_products['ma_san_pham'] == item_id].index[0]
                sim_scores = list(enumerate(cosine_sim_new[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:top_n+1]
                content_recommendations.extend([all_items[i[0]] for i in sim_scores])

        # 3. Tạo danh sách gợi ý từ collaborative filtering
        predictions = [model_svd.predict(user_id, item_id) for item_id in all_items]
        svd_scores = {str(pred.iid): pred.est for pred in predictions}
        cf_recommendations = sorted(svd_scores, key=svd_scores.get, reverse=True)[:top_n]

        # 4. Kết hợp hai danh sách và loại bỏ trùng lặp
        combined_recommendations = list(set(content_recommendations + cf_recommendations))

        # 5. Xếp hạng lại dựa trên điểm số từ cả hai mô hình
        hybrid_scores = {}
        content_scores_result = {}
        svd_scores_result = {}
        for item_id in combined_recommendations:
            content_score = 0
            if item_id in content_recommendations:
                content_score = 1 / (content_recommendations.index(item_id) + 1)
            cf_score = svd_scores.get(str(item_id), 0)
            hybrid_scores[item_id] = weight_content * content_score + (1 - weight_content) * cf_score
            content_scores_result[item_id] = content_score
            svd_scores_result[item_id] = cf_score

        # 6. Tạo DataFrame kết quả (đã thêm các cột từ df_products)
        recommendations_df = pd.DataFrame({
            'ma_san_pham': list(hybrid_scores.keys()),
            'ten_san_pham': [product_info.get(int(item_id), {}).get('ten_san_pham') for item_id in hybrid_scores.keys()],
            'hybrid_score': list(hybrid_scores.values()),
            'content_score': list(content_scores_result.values()),
            'svd_score': list(svd_scores_result.values()),
            'diem_trung_binh': [product_info.get(int(item_id), {}).get('diem_trung_binh') for item_id in hybrid_scores.keys()],
            'gia_ban': [product_info.get(int(item_id), {}).get('gia_ban') for item_id in hybrid_scores.keys()],
            'ty_le_giam_gia': [product_info.get(int(item_id), {}).get('ty_le_giam_gia') for item_id in hybrid_scores.keys()],
            'mo_ta': [product_info.get(int(item_id), {}).get('mo_ta') for item_id in hybrid_scores.keys()],
            'volume': [product_info.get(int(item_id), {}).get('volume') for item_id in hybrid_scores.keys()]
        })

    # Sắp xếp và trả về top_n sản phẩm (sửa lỗi UnboundLocalError)
    if not recommendations_df.empty:
        recommendations_df = recommendations_df.sort_values(by='hybrid_score', ascending=False).head(top_n)

    return recommendations_df

# Hiển thị nội dung tùy thuộc vào lựa chọn của người dùng
if selected_option == 'Mục tiêu - Nội dung':
    st.header("Mục Tiêu")
    # Hiển thị nội dung tùy thuộc vào lựa chọn của người dùng
    st.markdown(
        """
        <div style="font-size: 25px; text-align: justify;">
        Trong thời đại kỹ thuật số, việc cá nhân hóa trải nghiệm khách hàng là chìa khóa thành công cho các doanh nghiệp thương mại điện tử. Hiểu được điều này, chúng tôi đề xuất xây dựng một hệ thống gợi ý sản phẩm thông minh cho Hasaki, giúp khách hàng dễ dàng khám phá và lựa chọn những sản phẩm phù hợp với nhu cầu và sở thích. Bằng cách sử dụng các kỹ thuật Content-based filtering và Collaborative filtering, hệ thống sẽ phân tích dữ liệu sản phẩm và hành vi người dùng để đưa ra những gợi ý chính xác và hiệu quả, từ đó nâng cao trải nghiệm mua sắm và tăng doanh thu cho Hasaki.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.image("02.jpg")

    st.markdown(
        """
        <div style="font-size: 22px; text-align: justify;">
        Content-based filtering: Gợi ý sản phẩm dựa trên nội dung (thành phần, công dụng, thương hiệu,...) của sản phẩm mà người dùng đã quan tâm.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="font-size: 22px; text-align: justify;">
        Collaborative filtering: Gợi ý sản phẩm dựa trên hành vi mua hàng và đánh giá của những người dùng khác có sở thích tương tự.
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Chia thành 5 phần nhỏ
    st.header("Nội dung thực hiện")
    st.markdown("<span style='font-size: 20px; font-weight: bold;'>1. Khám phá dữ liệu</span>", unsafe_allow_html=True)
    with st.expander("click here"):
        st.write("Dữ liệu thu thập từ web Hasaki bao gồm:")
        st.write("- Dữ liệu sản phẩm gồm mẫ sản phẩm, tên sản phẩm, mô tả, thành phần, giá, số điểm trung bình")
        st.write("- Dữ liệu đánh giá gồm mã khách hàng, bình luận, ngày - giờ bình luận, số sao và mã sảm phẩm")
   # Chia phần 1 thành 5 mục nhỏ
        st.markdown(
        """
        <div style="font-size: 22px; font-weight: bold; text-align: justify;">
        1.1 Phân tích phân bổ số sao đánh giá
        </div>
        """,
        unsafe_allow_html=True,
    )
        
    # ... (Thêm nội dung cho mục 1.1) ...
        st.image('03_phantichsosao.jpg', use_container_width=True)
        st.write("Biểu đồ phân bố số sao cho thấy sự phân bố số sao đánh giá của sản phẩm, với số sao 5 chiếm tỷ lệ lớn nhất, cho thấy phần lớn khách hàng hài lòng với sản phẩm. Tuy nhiên, số sao 1 và 2 có vẻ ít phổ biến, điều này có thể cho thấy ít khách hàng không hài lòng hoặc có ít sản phẩm nhận đánh giá thấp. Mô hình Collaborative Filtering có thể gặp khó khăn khi xử lý sự mất cân bằng này, do đó cần phải cân nhắc các kỹ thuật để xử lý dữ liệu không đều.")

        st.markdown(
        """
        <div style="font-size: 22px; font-weight: bold; text-align: justify;">
        1.2 Tần suất bình luận theo giờ trong ngày
        </div>
        """,
        unsafe_allow_html=True,
    )
        
    # ... (Thêm nội dung cho mục 1.2) ...
        st.image('04_sobinhluantrongngay.jpg', use_container_width=True)
        st.write("Biểu đồ thể hiện tần suất bình luận theo giờ trong ngày, với số lượng bình luận cao nhất vào khoảng 20:00 đến 22:00, cho thấy người dùng có xu hướng bình luận vào buổi tối. Các giờ sáng sớm (0:00–6:00) có ít bình luận nhất, phản ánh hành vi người dùng ít hoạt động trong khoảng thời gian này. Điều này có thể cung cấp thông tin hữu ích cho việc tối ưu hóa thời gian đăng sản phẩm hoặc chạy chiến dịch marketing.")

        st.markdown(
        """
        <div style="font-size: 22px; font-weight: bold; text-align: justify;">
        1.3 Top 10 sản phẩm có nhiều bình luận nhất
        </div>
        """,
        unsafe_allow_html=True,
    )
        
    # ... (Thêm nội dung cho mục 1.3) ...
        st.image('05_sanphamconhieubinhluan.jpg', use_container_width=True)
        st.write("Biểu đồ top 10 sản phẩm có nhiều bình luận nhất thể hiện sự ưa chuộng rõ rệt của khách hàng đối với các sản phẩm tẩy trang của L'Oreal, điều này có thể được tận dụng trong Collaborative Filtering để tạo ra các gợi ý sản phẩm tương tự cho người dùng. Các sản phẩm như nước tẩy trang L'Oreal và mặt nạ Naruko có thể được đề xuất cho những người dùng đã mua hoặc đánh giá cao các sản phẩm tương tự. Việc phân tích số lượng bình luận và sự phổ biến của các sản phẩm giúp mô hình Collaborative Filtering đưa ra các gợi ý chính xác hơn, dựa trên sở thích và hành vi của người dùng.")

        st.markdown(
        """
        <div style="font-size: 22px; font-weight: bold; text-align: justify;">
        1.4 Nhóm giá bán phổ biến của sản phẩm
        </div>
        """,
        unsafe_allow_html=True,
    )
        
    # ... (Thêm nội dung cho mục 1.3) ...
        st.image('06_nhomgiabanphobien.jpg', use_container_width=True)
        st.write("Biểu đồ nhóm giá bán phổ biến của sản phẩm thể hiện phân bố giá bán của các sản phẩm trong dataset. Phần lớn sản phẩm có giá bán trong khoảng 58k - 431k, cho thấy đây là nhóm giá phổ biến nhất. Các nhóm giá cao hơn, như 1.28 triệu - 1.71 triệu và 1.71 triệu - 2.13 triệu, có tần suất thấp hơn, phản ánh rằng các sản phẩm có giá cao hơn ít phổ biến hơn.")

        st.markdown(
        """
        <div style="font-size: 22px; font-weight: bold; text-align: justify;">
        1.5 Phân tích điểm trung bình của sản phẩm
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ... (Thêm nội dung cho mục 1.3) ...
        st.image('07_diemtrungbinhsanpham.jpg', use_container_width=True)
        st.write("Biểu đồ điểm trung bình của sản phẩm thể hiện phân bố điểm trung bình của các sản phẩm trong dataset. Điểm trung bình chủ yếu tập trung vào khoảng 3 đến 5 điểm, cho thấy đa số sản phẩm nhận được đánh giá khá tốt từ khách hàng. Một số ít sản phẩm có điểm trung bình thấp hơn, phản ánh sự không hài lòng hoặc vấn đề về chất lượng từ người dùng.")

    st.markdown("<span style='font-size: 20px; font-weight: bold;'>2. Tiền xử lý dữ liệu</span>", unsafe_allow_html=True)
    with st.expander("click here"):
        st.markdown(
        """
        <div style="font-size: 22px; font-weight: bold; text-align: justify;">
        Tiền xử lý dữ liệu sản phẩm
        </div>
        """,
        unsafe_allow_html=True,
    )    
        
        st.image('08_preprocessingsanpham.jpg', use_container_width=True)
        st.markdown(
        """
        <div style="font-size: 22px; font-weight: bold; text-align: justify;">
        Dữ liệu sau khi tiền xử lý
        </div>
        """,
        unsafe_allow_html=True,
    )       
        
        st.image('09_sanpham.jpg', use_container_width=True)
        st.image('10_sanphamwordcloud.png', use_container_width=True)
    st.markdown("<span style='font-size: 20px; font-weight: bold;'>3. Xây dựng cosine_similarity (Content-based Filtering)</span>", unsafe_allow_html=True)
    with st.expander("click here"):

        st.write("Cosine Similarity là một phương pháp đo lường độ tương đồng giữa hai vector. Trong trường hợp này, mỗi sản phẩm sẽ được biểu diễn dưới dạng một vector đặc trưng, và Cosine Similarity sẽ tính toán góc giữa hai vector để xác định độ tương đồng.")
        st.write("Giá trị Cosine Similarity càng gần 1 thì hai sản phẩm càng giống nhau.")
        st.markdown(
        """
        <div style="font-size: 17px; font-weight: bold; text-align: justify;">
        Quy trình xây dựng mô hình Cosine Similarity
        </div>
        """,
        unsafe_allow_html=True,
    )
        st.image('11_cosin.jpg', use_container_width=True)
        st.markdown(
        """
        <div style="font-size: 17px; font-weight: bold; text-align: justify;">
        Ma trận độ tương đồng cosine (Cosine Similarity Matrix) giữa 10 sản phẩm
        </div>
        """,
        unsafe_allow_html=True,
    )
        
        st.image('12_cosintrucquanhoa.png', use_container_width=True)
        st.write("Heatmap thể hiện độ tương đồng cosine giữa 10 sản phẩm.  Giá trị càng cao (gần 1) màu càng đậm, thể hiện độ tương đồng lớn. Sản phẩm 1 và 9 rất giống nhau (0.92), trong khi sản phẩm 0 và 3 khác biệt (0.01). Heatmap giúp  nhận biết nhóm sản phẩm tương đồng và  đánh giá hiệu quả của thuật toán.")

        st.markdown(
        """
        <div style="font-size: 17px; font-weight: bold; text-align: justify;">
        Ưu điểm của Cosine Similarity
        </div>
        """,
        unsafe_allow_html=True,
    )
        st.write("Cosine Similarity là một phương pháp đơn giản, hiệu quả và dễ dàng triển khai, hoạt động tốt trong việc tìm kiếm các sản phẩm tương tự dựa trên nội dung.")
    st.markdown("<span style='font-size: 20px; font-weight: bold;'>4. Xây dựng Surprise (Collaborative Filtering)</span>", unsafe_allow_html=True)
    with st.expander("click here"):
        st.write("Collaborative Filtering có thể khám phá ra những mối liên hệ tiềm ẩn giữa người dùng và sản phẩm mà Content-Based Filtering không thể, nó có thể gợi ý những sản phẩm mới hoặc ít phổ biến mà người dùng chưa từng biết đến.")
        st.write("Surprise là một thư viện Python mã nguồn mở được thiết kế dành riêng cho việc xây dựng và phân tích các hệ thống gợi ý, đặc biệt là CF.  Nó cung cấp một bộ công cụ toàn diện và dễ sử dụng, cho phép:")
        st.write("- Xử lý dữ liệu đánh giá từ nhiều nguồn khác nhau.")
        st.write("- Triển khai các thuật toán CF phổ biến như KNN, SVD, ...")
        st.write("- Đánh giá hiệu quả của mô hình với các độ đo như RMSE, MAE.")
        st.write("- Tinh chỉnh siêu tham số để tối ưu hóa hiệu suất.")
        
        st.write("SVD là một kỹ thuật mạnh mẽ được ứng dụng rộng rãi trong nhiều lĩnh vực như xử lý ảnh, nén dữ liệu và đặc biệt là trong các hệ thống gợi ý. Ứng dụng của SVD có thể kể đến như dự đoán điểm đánh giá của người dùng cho các sản phẩm và gợi ý những sản phẩm có điểm số cao, tìm các sản phẩm có đặc trưng tiềm ẩn giống nhau và gom nhóm người dùng có sở thích tương tự nhau.")
        st.markdown(
        """
        <div style="font-size: 17px; font-weight: bold; text-align: justify;">
        Quy trình xây dựng mô hình SVD
        </div>
        """,
        unsafe_allow_html=True,)
        st.image('13_svd.jpg', use_container_width=True)
        st.markdown("<span style='font-size: 16px; font-weight: bold;'>Ưu điểm của SVD</span>", unsafe_allow_html=True)
        st.write("Hoạt động tốt với dữ liệu thưa thớt, thường gặp trong các hệ thống gợi ý, nơi người dùng chỉ đánh giá một số ít sản phẩm.")
        st.write("Xử lý được số lượng người dùng và sản phẩm lớn.")
        st.write("Cho kết quả dự đoán khá chính xác.")
        st.markdown("<span style='font-size: 16px; font-weight: bold;'>Kết luận</span>", unsafe_allow_html=True)
        st.write("Collaborative Filtering, với sự hỗ trợ của thư viện Surprise và thuật toán SVD, là một công cụ mạnh mẽ để xây dựng các hệ thống gợi ý cá nhân hóa, giúp nâng cao trải nghiệm người dùng và mang lại hiệu quả kinh doanh cho doanh nghiệp.")

    st.markdown("<span style='font-size: 20px; font-weight: bold;'>5. Kết hợp content-based filtering và Collaborative filtering </span>", unsafe_allow_html=True)
    with st.expander("click here"):
        st.markdown(
        """
        <div style="font-size: 17px; font-weight: bold; text-align: justify;">
        a)	Tạo danh sách gợi ý từ mô hình content-based: 
        </div>
        """,
        unsafe_allow_html=True,
    )    
        st.write("Sử dụng ma trận cosine_sim để tìm ra top N sản phẩm tương tự với những sản phẩm người dùng đã đánh giá cao.")
    
        st.markdown(
        """
        <div style="font-size: 17px; font-weight: bold; text-align: justify;">
        b)	Tạo danh sách gợi ý từ mô hình collaborative filtering: 
        </div>
        """,
        unsafe_allow_html=True,
    )   
        st.write("Sử dụng mô hình SVD (model_svd) để dự đoán xếp hạng và chọn ra top N sản phẩm có xếp hạng dự đoán cao nhất.")
    
        st.markdown(
        """
        <div style="font-size: 17px; font-weight: bold; text-align: justify;">
        c)	Kết hợp hai danh sách: 
        </div>
        """,
        unsafe_allow_html=True,
    )   
        st.write("- Gộp hai danh sách và loại bỏ các sản phẩm trùng lặp.")
        st.write("-	Xếp hạng các sản phẩm trong danh sách kết hợp dựa trên thứ tự xuất hiện trong hai danh sách ban đầu (ví dụ: sản phẩm xuất hiện ở vị trí cao hơn trong cả hai danh sách sẽ được xếp hạng cao hơn).")
        st.write("-	Chọn ra top K sản phẩm từ danh sách kết hợp để gợi ý cho người dùng.")
            
elif selected_option == 'Cosine':
    st.header("content-based filtering- cosine_similarity")
    st.write("-----------------------------------------------")

    # Hiển thị bảng dữ liệu sản phẩm
    st.dataframe(df_products) 

    # Phần chọn sản phẩm và hiển thị sản phẩm liên quan
    st.header("Chọn sản phẩm")

    if 'selected_ma_san_pham' not in st.session_state:
        st.session_state.selected_ma_san_pham = None

    product_options = [(row['ten_san_pham'], row['ma_san_pham']) for index, row in df_products.iterrows()]

    selected_product = st.selectbox(
        "Chọn sản phẩm",
        options=product_options,
        format_func=lambda x: x[0]
    )

    st.write("Bạn đã chọn:", selected_product)
    st.session_state.selected_ma_san_pham = selected_product[1]

    if st.session_state.selected_ma_san_pham:
        st.write("ma_san_pham: ", st.session_state.selected_ma_san_pham)
        selected_product_df = df_products[df_products['ma_san_pham'] == st.session_state.selected_ma_san_pham]

        if not selected_product_df.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_product_df['ten_san_pham'].values[0])

            product_description = selected_product_df['mo_ta'].values[0]
            truncated_description = ' '.join(product_description.split()[:100])
            st.write('##### Thông tin:')
            st.write(truncated_description, '...')

            st.write('##### Các sản phẩm liên quan:')
            recommendations = get_recommendations(df_products, st.session_state.selected_ma_san_pham, cosine_sim=cosine_sim_new, nums=3) 
            display_recommended_products(recommendations, cols=3)
        else:
            st.write(f"Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}")       

elif selected_option == 'Surprise':
    st.header("colabrative filtering- surprise")
    st.write("-----------------------------------------------")

    # Tạo text_input để nhập liệu
    user_id_input = st.text_input("Nhập User ID mới (giá trị lớn hơn 4700):")

    # Tạo selectbox với danh sách User ID
    st.write('Lưu ý: Xóa user ID mới trước khi vào chọn các User ID cũ')
    user_id_options = danh_gia2['ma_khach_hang'].unique().tolist()
    selected_user_id = st.selectbox("Chọn User ID cũ:", user_id_options)

    # Kiểm tra xem người dùng đã nhập liệu hay chọn từ danh sách
    if user_id_input:
        user_id = int(user_id_input)
        if is_new_user(user_id):
            st.write(user_id, " là người dùng mới, các sản phẩm gợi ý:")
            recommendations = recommend_products_for_new_user_with_svd(user_id, model_svd, df_products, top_n=10)
        else:
            st.write(user_id, " là người dùng cũ, các sản phẩm gợi ý:")
            recommendations = recommend_products_for_old_user(user_id, model_svd, st.session_state.danh_gia2, df_products, top_n=10)
        st.dataframe(recommendations)

        # Hiển thị kết quả
        if not recommendations.empty:  # Kiểm tra recommendations.empty
            st.write("Các sản phẩm gợi ý cho người dùng này")
            display_recommended_products(recommendations.head(3))  # Gọi hàm và truyền recommendations.head(3)

    elif selected_user_id:
        user_id = selected_user_id
        if is_new_user(user_id):
            st.write(user_id, " là người dùng mới, các sản phẩm gợi ý:")
            recommendations = recommend_products_for_new_user_with_svd(user_id, model_svd, df_products, top_n=10)
        else:
            st.write(user_id, " là người dùng cũ, các sản phẩm gợi ý:")
            recommendations = recommend_products_for_old_user(user_id, model_svd, st.session_state.danh_gia2, df_products, top_n=10)
        st.dataframe(recommendations)

        # Hiển thị kết quả
        if not recommendations.empty:  # Kiểm tra recommendations.empty
            st.write("Các sản phẩm gợi ý cho người dùng này")
            display_recommended_products(recommendations.head(3))  # Gọi hàm và truyền recommendations.head(3)

elif selected_option == 'Hybrid':
    st.header("Kết hợp content-based filtering và Collaborative filtering")
    st.write("-----------------------------------------------")

    # Lấy danh sách user_id từ dữ liệu đánh giá
    user_ids = danh_gia['ma_khach_hang'].unique().tolist()

    # Tạo text_input để nhập liệu
    user_id_input = st.text_input("Nhập User ID mới (giá trị lớn hơn 4700):")

    # Tạo selectbox với danh sách User ID
    st.write('Lưu ý: Xóa user ID mới trước khi vào chọn các User ID cũ')
    selected_user_id = st.selectbox("Chọn User ID cũ:", user_ids)

    # Kiểm tra xem người dùng đã nhập liệu hay chọn từ danh sách
    if user_id_input:
        user_id = int(user_id_input)
    else:
        user_id = selected_user_id

    if user_id:
        try:
            recommendations_df = simple_hybrid_recommendation(
                user_id,
                danh_gia,
                st.session_state.df_products,
                cosine_sim_new,
                model_svd
            )

            # Kiểm tra người dùng mới hay cũ và hiển thị kết quả
            if check_if_new_user(user_id, danh_gia, st.session_state.khach_hang):
                st.write(user_id, " là người dùng mới.")
            else:
                st.write(user_id, " là người dùng cũ.")

            st.write("Sản phẩm gợi ý:")
            st.dataframe(recommendations_df)

            # Hiển thị kết quả
            if not recommendations_df.empty:
                st.write("Các sản phẩm gợi ý cho người dùng này")

                # Giới hạn 3 sản phẩm đầu
                recommendations_df = recommendations_df.head(3)

                # Tạo 3 cột
                col1, col2, col3 = st.columns(3)

                # CSS để hiển thị các cột theo hàng ngang
                st.markdown(
                    """
                    <style>
                    .stColumns {
                        flex-direction: row !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                # Hiển thị sản phẩm
                for i, col in enumerate([col1, col2, col3]):
                    with col:
                        product = recommendations_df.iloc[i]
                        st.write(product['ten_san_pham'])
                        st.write(f"Giá bán: {product['gia_ban']}")
                        st.write(f"Giảm giá: {product['ty_le_giam_gia']:.0f}%")
                        st.write(f"Điểm trung bình: {product['diem_trung_binh']}")
                        st.write(f"Volume: {product['volume']}")

                        with st.expander("Mô tả"):
                            st.write(product['mo_ta'])

        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")