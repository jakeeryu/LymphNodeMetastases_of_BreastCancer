# LymphNodeMetastases_of_BreastCancer
Lymph Node Metastases of Breast Cancer
<br/><hr/><br/>
### 프로젝트 주제
- 유방암 병리 슬라이드 영상과 임상 항목을 통한 유방암의 임파선 전이 여부 이진 분류 예측

### 프로젝트 내용
- Tabular data, Image Data에서 각각 특징을 추출하고 합쳐, 하나의 딥러닝 모델으로 사용하여 이진분류

### 데이터
- https://dacon.io/competitions/official/236011/data <br/>
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


<br/><hr/><br/>
### 데이터 설명
> 데이터는 위 URL에서 다운로드 받아야 합니다.
<br/>

#### 1. Image Data

> 유방암 환자의 병리 슬라이드 이미지<br/>
> Image Max Height 8299 / Image max Width 3991<br/>
(유방암 환자의 조직을 염색하여 동결절편한 것의 이미지입니다.)

- p_images
> Image에 Padding을 적용하여 고정 Size로 설정한 Image 입니다.

- 1024_folder
> Image의 가로/세로 크기 중 큰 값을 가지고 비율로 설정하여 Max 1024에 맞추어 Resize 한 Image 입니다.<br/>
> 이미지 손실을 줄이기 위하여 INTER_AREA, INTER_LANCZOS4 두가지의 보간법을 사용하였습니다.

- FULL_PADDING
> Image의 Max Width, Height 값을 가지고 Padding한 데이터 입니다.

<br/>

#### 2. Tabuler Data

> 유방암 환자의 전이여부 검사 결과 항목과 환자 개인정보, 병리학 이미지 경로 등이 입력된 28개 column의 데이터<br/>
> 28 columns : ID, img_path, mask_path(Just train set), 나이, 수술연월일, 진단명, 암의위치, 암의개수, 암의장경, NG, HG, HG_score1, HG_score2, HG_score3, DCIS_or_LCIS 여부, DCIS_or_LCIS_type, T_category, ER, ER_Allred_score, PR, PR_Allred_score, KI-67_LI_percent, HER2, HER2_IHC, HER2_SISH, HER2_SISH_ratio, BRCA_mutation, N_category(Just train set)


<br/><hr/><br/>
### Code File 설명
- 1. File Name 00_
> Mask Image를 확인
- 2. File Name 01_
> EDA
- 3. File Name 02_
> 제거
- 4. File Name 03_
> Activation Function에 따른 Score를 확인
- 5. File Name 04_
> Model 구성 및 변경


<br/><hr/><br/>
### Activate Function Validation Score
<table>
  <tr>
    <th>Activation</th>
    <th>Last Epoch</th>
    <th>Max Achieved</th>
    <th>Average</th>
  </tr>
  
  <tr>
    <th>LeakyReLU [Baseline]</th>
    <th>77.892</th>
    <th>77.892</th>
    <th>74.228</th>
  </tr>
  
  <tr>
    <th>LeakyReLU (Reduced)</th>
    <th>75.758</th>
    <th>76.540</th>
    <th>75.286</th>
  </tr>
  
  <tr>
    <th>ELU</th>
    <th>75.706</th>
    <th><strong>80.390</strong></th>
    <th>75.958</th>
  </tr>
  
  <tr>
    <th>ELU (Added)</th>
    <th>76.768</th>
    <th>79.884</th>
    <th>77.997</th>
  </tr>
  
  <tr>
    <th>GELU</th>
    <th>77.249</th>
    <th>77.679</th>
    <th>74.743</th>
  </tr>
  
  <tr>
    <th>GELU (Reduced)</th>
    <th>75.485</th>
    <th>77.422</th>
    <th>75.806</th>
  </tr>
  
  <tr>
    <th>GELU (Added)</th>
    <th>78.304</th>
    <th>78.435</th>
    <th>76.896</th>
  </tr>
  
  <tr>
    <th>SELU (Added)</th>
    <th>72.806</th>
    <th>74.960</th>
    <th>73.193</th>
  </tr>
</table>



<br/><hr/><br/>
<details>
  <summary>Reference</summary>
  Multiple Instance Learning: Model Pipeline<br/>
  https://www.youtube.com/watch?v=h5qThVdAfOQ<br/><br/>
  Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning<br/>
  https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.pdf<br/><br/>
  Multiple Instance Learning<br/>
  https://medium.com/swlh/multiple-instance-learning-c49bd21f5620<br/><br/>
  MedAI #36: Weakly supervised tumor detection in whole slide image analysis | Bin Li<br/>
  https://www.youtube.com/watch?v=ZPe94q8wxPQ<br/><br/>
  2018 Data Science Bowl: Find the nuclei in divergent images to advance medical discovery<br/>
  https://www.kaggle.com/competitions/data-science-bowl-2018/data<br/><br/>
  Breast Cancer Classification With PyTorch and Deep Learning<br/>
  https://medium.com/swlh/breast-cancer-classification-with-pytorch-and-deep-learning-52dd62362157<br/><br/>
  Deep learning-based cross-classifications reveal conserved spatial behaviors within tumor histological images<br/>
  https://www.nature.com/articles/s41467-020-20030-5<br/><br/>
  Deep Learning Models for Histopathological Classification of Gastric and Colonic Epithelial Tumours<br/>
  https://www.nature.com/articles/s41598-020-58467-9<br/><br/>
  Finetuning Torchvision Models<br/>
  https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#inception-v3<br/><br/>
  Torchvision.models<br/>
  https://pytorch.org/vision/0.8/models.html<br/><br/>
  U-Net: Convolutional Networks for Biomedical Image Segmentation<br/>
  https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/<br/><br/>
  U-Net For Brain MRI<br/>
  https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/<br/><br/>
  Pytorch에서 학습한 모델 저장 및 불러오기<br/>
  https://justkode.kr/deep-learning/pytorch-save<br/><br/>
  Models and Pre-trained Weights<br/>
  https://pytorch.org/vision/stable/models.html<br/><br/>
  U-Net Model with submission<br/>
  https://www.kaggle.com/code/hmendonca/u-net-model-with-submission/notebook<br/><br/>
  
</details>
