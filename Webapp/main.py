import streamlit as st
import pickle as pkl
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

def get_clean_data():
    df = pd.read_excel('data/Data.xlsx')
    data = df.iloc[:, 1:]
    data.info()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].apply(lambda x: str(x).split('-')[0])
            data[col] = data[col].apply(lambda x: str(x).split('.')[0])
    return data

def add_sidebar():
    st.sidebar.header("Xin hãy điền thông tin khách hàng cá nhân")
    
    slider_labels = [
        ("Giới tính:", "Gender"),
        ("Độ Tuổi của khách hàng ", "Age"),
        ("Mức lương hàng tháng (Triệu VND)", "Income"),
        ("Khách hàng có tài sản đảm bảo không ?", "Collateral"),
        ("Tình trạng nơi ở của khách hàng ? ", "House_Status"),
        ("Tình trạng hôn nhân của khách hàng ?", "Marriage"),
        ("Thời hạn khách hàng muốn vay (Theo tháng):", "Duration"),
        ("Lịch sử khoản vay trước của khách hàng :", "Loan History"),
        ("Mục đích khách hàng vay là gì?", "Loan Purpose"),
      ]
        
    input_dict = {}

    for label, key in slider_labels:
      if key == "Gender":
        Gen_display = ('Nam', 'Nữ')
        gen_option = list(range(len(Gen_display)))
        input_dict[key] = st.sidebar.radio(label, gen_option, format_func= lambda x: Gen_display[x])
      elif key == "Age":
        input_dict[key] = st.sidebar.number_input(label, min_value=18, max_value=100)
      elif key == "Marriage":
        mar_display = ("Độc thân","Đã kết hôn","Ly Hôn")
        mar_option = list(range(len(mar_display)))
        input_dict[key] = st.sidebar.selectbox(label, mar_option, format_func= lambda x: mar_display[x],index=0)
      elif key == "Income":
        input_dict[key] = st.sidebar.number_input(label, min_value=0)
      elif key == "Collateral":
        collateral_display = (
          "Không có tài sản đảm bảo",
          "Có 1 phần tài sản đảm bảo",
          "Có đủ tài sản đảm bảo được bảo lãnh",
          "Có đủ tài sản đảm bảo của bản thân")
        col_option = list(range(len(collateral_display)))
        input_dict[key] = st.sidebar.selectbox(label, col_option, format_func= lambda x: collateral_display[x],index=0)
      elif key == "Loan Purpose":
        loanpurpose_display = ("Vay phục vụ nhu cầu đời sống","Vay phục vụ SXKD", "Vay mục đích mua bán nhà đất","Vay để xây nhà và sửa chữa nhà")
        loan_op = list(range(len(loanpurpose_display)))
        input_dict[key] = st.sidebar.selectbox(label, loan_op,format_func= lambda x: loanpurpose_display[x],index=0)
      elif key == "House_Status":
        housingstatus_display = ("Nhà thuê","Sở hữu nhà riêng","Đang thế chấp nhà")
        house_op = list(range(len(housingstatus_display)))
        input_dict[key] = st.sidebar.selectbox(label, house_op, format_func= lambda x: housingstatus_display[x], index=0)
      elif key == "Duration":
        input_dict[key] = st.sidebar.number_input(label, min_value=0)
      elif key == "Loan History":
        loanhistory_display = ("Trả đúng hạn","Trả trễ hạn")
        loanhis_op = list(range(len(loanhistory_display)))
        input_dict[key] = st.sidebar.selectbox(label, loanhis_op,format_func=  lambda x: loanhistory_display[x], index=0)
    return input_dict
  

def get_scaled_values():
    data = get_clean_data()
    X = data.drop(['Credithworthiness'], axis=1)
    feature_cols = ['Gender','Age','Income','Collateral','House_Status','Marriage','Duration','Loan History','Loan Purpose']
    X = data[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def add_predictions(feature):
  model = pkl.load(open("train_model/model.pkl", "rb"))
  scaler = pkl.load(open("train_model/scaler.pkl", "rb"))
  

  input_array_scaled = scaler.transform(feature)
  
  prediction = model.predict(input_array_scaled)
  
  st.subheader("PREDICTION")
  st.write("Kết quả: Khách hàng có khả năng trả nợ")
  
  if prediction == 0:
    st.write("<span class='repayment ontime'>ĐÚNG HẠN</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='repayment late'>TRỄ HẠN</span>", unsafe_allow_html=True)


  st.write("Xác suất kết quả trả nợ Đúng Hạn: ", model.predict_proba(input_array_scaled)[0][0])
  st.write("Xác suất kết quả trả nợ Trễ Hạn: ", model.predict_proba(input_array_scaled)[0][1])
  
  
def main():
  st.set_page_config(
    page_title="Loan Repayment Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
  )

  with open("assets/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
  with st.container():
    st.title("Loan Repayment Predictor")
    st.write("by Khanh Huyen")
    st.subheader('Xác nhận thông tin khách hàng như sau: ')

  input_data = add_sidebar()
  feature = pd.DataFrame(input_data, index=[0])

  
  if 'button_clicked' not in st.session_state:
      st.session_state.button_clicked = False 
    
  st.write(feature)
    
  if st.button('Result'):
    st.session_state.button_clicked = True
    if  st.session_state.button_clicked:
        add_predictions(feature)
  st.write()
if __name__ == '__main__':
  main()