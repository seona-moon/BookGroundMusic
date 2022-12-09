# 🎶 BookGroundMusic 
### : 스타트 21팀 백조🦢

저희 팀의 BGM, 북그라운드뮤직은 전자책 텍스트 분석을 통해 실시간으로 분위기에 맞는 배경음악을 제공해주는 백그라운드 어플입니다.

기술 구현 흐름은 다음과 같습니다.  
<a href="#"><img src="https://user-images.githubusercontent.com/96529879/206640737-9b0ab936-f5da-4319-9c57-5072a878c386.png" width="450px" alt="sample image"></a>  

가장 먼저 스크린 OCR로 화면 속 텍스트 데이터를 수집하여, 이를 텍스트 전처리 모델에 넘겨주어 감성분석을 진행합니다.  
이 분석 결과를 음악 감성분석 결과와 매치시켜 최종적으로 매핑된 음악을 사용자에게 제공하게 됩니다.

📁 multiclass-emotion-classifier  
📁 multiclass-
