# LymphNodeMetastases_of_BreastCancer
Lymph Node Metastases of Breast Cancer

### 프로젝트 주제
- 유방암 병리 슬라이드 영상과 임상 항목을 통한 유방암의 임파선 전이 여부 이진 분류 예측

### 프로젝트 내용
- Tabular data, Image Data에서 각각 특징을 추출하고 합쳐, 하나의 딥러닝 모델으로 사용하여 이진분류

### 데이터
- https://dacon.io/competitions/official/236011/data
데이터의 모든 권한은 데이콘에 있습니다. (DataSet Is Copyright DACON Inc. All rights reserved)

### 개발환경
- Apple Silicon MacMini CTO, MacBook Pro
- Python 3.8, Pytorch, OpenCV, Sklearn, Pandas, Etc..
- Efficientnet_b0 pretrained

### 프로젝트 기대효과
- 임파선은 암의 전이에 치명적인 역할을 하므로, 림프절 전이 여부에 따라 치료와 예후가 크게 좌우된다.
- 따라서 림프절 전이 여부와 전이 단계를 파악하는 것이 암의 치료와 진단에 매우 중요하므로 이에 기여한다.

### 팀 구성
<table>
  <tr>
    <th>이름</th>
    <th>역할</th>
    <th>담당업무</th>
    <th>공동업무</th>
  </tr>
  
  <tr>
    <th>권진욱</th>
    <th>팀장</th>
    <th>Image Processing / EDA</th>
    <th rowspan="4">Modeling</th>
  </tr>
  
  <tr>
    <th>류제욱</th>
    <th>팀원</th>
    <th>Tabular Data Feature Extraction</th>
  </tr>
  
  <tr>
    <th>전대광</th>
    <th>팀원</th>
    <th>Tabular Data EDA / Wrangling</th>
  </tr>
  
  <tr>
    <th>송일수</th>
    <th>팀원</th>
    <th>Reference / paper searching</th>
  </tr>
</table>

### 데이터 설명
1. Image Data
- 유방암 환자의 병리 슬라이드 이미지
- Image Max Height 8299 / Image max Width 3991
(유방암 환자의 조직을 염색하여 동결절편한 것의 이미지입니다.)

> p_images
- Image에 Padding을 적용하여 고정 Size로 설정한 Image 입니다.

> r_images
- Image의 가로/세로 크기 중 큰 값을 가지고 비율로 설정하여 Max 1024에 맞추어 Resize 한 Image 입니다.
- 이미지 손실을 줄이기 위하여 INTER_AREA, INTER_LANCZOS4 두가지의 보간법을 사용하였습니다.

2. Tabuler Data
- 유방암 환자의 전이여부 검사 결과 항목과 환자 개인정보, 병리학 이미지 경로 등이 입력된 28개 column의 데이터
- 28 columns : ID, img_path, mask_path(Just train set), 나이, 수술연월일, 진단명, 암의위치, 암의개수, 암의장경, NG, HG, HG_score1, HG_score2, HG_score3, DCIS_or_LCIS 여부, DCIS_or_LCIS_type, T_category, ER, ER_Allred_score, PR, PR_Allred_score, KI-67_LI_percent, HER2, HER2_IHC, HER2_SISH, HER2_SISH_ratio, BRCA_mutation, N_category(Just train set)



### Code File 설명
1. filename
- 설명
2. filename
- 설명



