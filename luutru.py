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
        <div style="background-color: #f1f0f5; padding: 10px; border-radius: 5px;">
            <h5>Họ và tên nhóm 4:</h5>
            <p style='font-size: 24px;'>1. Phan Thị Thu Hạnh</p>  
            <p style='font-size: 24px;'>2. Nguyễn Hải Yến</p>  
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    st.write(f"<p style='font-size: {font_size}px;'>Giáo viên hướng dẫn: Khuất Phương Thùy</p>", unsafe_allow_html=True)
    font_size = 24
    st.write(f"<p style='font-size: {font_size}px;'>Ngày báo cáo: 14/12/2024</p>", unsafe_allow_html=True)

# Tiêu đề trang
st.title("Menu")

# Tạo menu sổ xuống (Dropdown)
menu_options = ['Mục tiêu', 'Cosine', 'Surprise']
# Tạo st.selectbox
with st.container():
    st.markdown(
        "<p style='font-size: 20px; font-weight: bold; margin-bottom: 5px;'>Chọn một mục:</p>",  # Giảm margin-bottom
        unsafe_allow_html=True
    )
    selected_option = st.selectbox('--', menu_options)

# Các hàm cho surprise
# --- Hàm recommend_products_svd ---
def recommend_products_svd(user_id, model, danh_gia, san_pham, top_n=5):
    """
    Hàm gợi ý sản phẩm cho người dùng dựa trên mô hình SVD.

    Args:
        user_id (int): ID của người dùng.
        model: Mô hình SVD đã được huấn luyện.
        danh_gia (pd.DataFrame): DataFrame chứa dữ liệu đánh giá.
        san_pham (pd.DataFrame): DataFrame chứa thông tin sản phẩm.
        top_n (int): Số lượng sản phẩm muốn gợi ý (mặc định là 5).

    Returns:
        pd.DataFrame: DataFrame chứa thông tin của top_n sản phẩm được gợi ý.
    """

    # Lấy danh sách tất cả các sản phẩm
    df_score = danh_gia[["item_id"]].drop_duplicates()

    # Dự đoán điểm đánh giá cho từng sản phẩm
    df_score['EstimateScore'] = df_score['item_id'].apply(lambda x: model.predict(user_id, int(x)).est)

    # Sắp xếp theo EstimateScore giảm dần
    df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)

    # Tạo dictionary ánh xạ giữa item_id và ten_san_pham
    item_name_mapping = dict(zip(san_pham['ma_san_pham'], san_pham['ten_san_pham']))

    # Thêm cột tên sản phẩm
    df_score['ten_san_pham'] = df_score['item_id'].map(item_name_mapping)

    # Hiển thị top_n sản phẩm gợi ý (bao gồm tên sản phẩm)
    return df_score[['item_id', 'ten_san_pham', 'EstimateScore']].head(top_n)

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
                    st.write(f"Điểm trung bình: {product['diem_trung_binh']}")
                    st.write(f"Volume: {product['volume']}")
                    
                    expander = st.expander(f"Mô tả")
                    product_description = product['mo_ta']
                    truncated_description = ' '.join(product_description.split()[:100]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")           

# Đọc dữ liệu sản phẩm
df_products = pd.read_csv('GUIthird_san_pham_processed_3nd.csv')

# Open and read file to cosine_sim_new
with open('products_cosine_sim.pkl', 'rb') as f:
    cosine_sim_new = pickle.load(f)

# Hiển thị nội dung tùy thuộc vào lựa chọn của người dùng
if selected_option == 'Mục tiêu':
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
    st.image("1.jpg")

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
    # Chia thành 4 phần nhỏ
    
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
        st.image('phantichsosao.jpg', use_container_width=True)
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
        st.image('sobinhluantrongngay.jpg', use_container_width=True)
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
        st.image('sanphamconhieubinhluan.jpg', use_container_width=True)
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
        st.image('nhomgiabanphobien.jpg', use_container_width=True)
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
        st.image('diemtrungbinhsanpham.jpg', use_container_width=True)
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
        
        st.image('preprocessingsanpham.jpg', use_container_width=True)
        st.markdown(
        """
        <div style="font-size: 22px; font-weight: bold; text-align: justify;">
        Dữ liệu sau khi tiền xử lý
        </div>
        """,
        unsafe_allow_html=True,
    )       
        
        st.image('sanpham.jpg', use_container_width=True)
        st.image('sanphamwordcloud.png', use_container_width=True)
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
        st.image('cosin.jpg', use_container_width=True)
        st.markdown(
        """
        <div style="font-size: 17px; font-weight: bold; text-align: justify;">
        Ma trận độ tương đồng cosine (Cosine Similarity Matrix) giữa 10 sản phẩm
        </div>
        """,
        unsafe_allow_html=True,
    )
        
        st.image('cosintrucquanhoa.png', use_container_width=True)
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
        st.image('svd.jpg', use_container_width=True)
        st.markdown("<span style='font-size: 16px; font-weight: bold;'>Ưu điểm của SVD</span>", unsafe_allow_html=True)
        st.write("Hoạt động tốt với dữ liệu thưa thớt, thường gặp trong các hệ thống gợi ý, nơi người dùng chỉ đánh giá một số ít sản phẩm.")
        st.write("Xử lý được số lượng người dùng và sản phẩm lớn.")
        st.write("Cho kết quả dự đoán khá chính xác.")
        st.markdown("<span style='font-size: 16px; font-weight: bold;'>Kết luận</span>", unsafe_allow_html=True)
        st.write("Collaborative Filtering, với sự hỗ trợ của thư viện Surprise và thuật toán SVD, là một công cụ mạnh mẽ để xây dựng các hệ thống gợi ý cá nhân hóa, giúp nâng cao trải nghiệm người dùng và mang lại hiệu quả kinh doanh cho doanh nghiệp.")

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
    st.header("Collaborative filtering - Surprise")
    st.write("-----------------------------------------------")

    # --- Phần xử lý cho SVD ---

    # Tải mô hình SVD đã lưu
    with open('svd_model.pkl', 'rb') as f:
        model_svd = pickle.load(f)

    # Tải dữ liệu đánh giá (Rating data)
    danh_gia = pd.read_csv('Danh_gia_filtered.csv')

    # Đổi tên cột cho phù hợp
    danh_gia.rename(columns={
        'ma_khach_hang': 'user_id',
        'ma_san_pham': 'item_id',
        'so_sao': 'rating'
    }, inplace=True)

    # Tạo Reader để đọc dữ liệu cho SVD
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(danh_gia[['user_id', 'item_id', 'rating']], reader)

    # Lấy danh sách user_id từ dữ liệu đánh giá
    user_ids = danh_gia['user_id'].unique().tolist()

# Chọn người dùng từ danh sách
    user_id = st.selectbox('Chọn ID người dùng:', user_ids)

# Gợi ý sản phẩm
    if st.button('Gợi ý sản phẩm'):
        recommendations = recommend_products_svd(user_id, model_svd, danh_gia, df_products)

    # Hiển thị kết quả
        st.write("Các sản phẩm gợi ý cho người dùng này:")
        st.write(recommendations)
