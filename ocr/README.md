## OCR을 통한 텍스트 데이터 수집

✔️ 전자책의 텍스트 관련하여 텍스트 파일을 직접 소지하고 있을 시 저작권 문제가 우려되어, 사용자가 기기를 통해 전자책을 읽을 때 해당 화면을 실시간으로 읽어와 **OCR로 분석**함으로써 텍스트 데이터를 수집하려고 합니다.  

  
### Tesseract 설치  
OCR Framework 중 한글 인식도와 어플리케이션 개발 시 활용성을 고려하여 **Tesseract**로 프레임워크를 선정하였습니다. 이 프로젝트에서는 한글책만을 대상으로 하여 한글 인식의 정확도만을 고려하였습니다 :)  

아래의 깃허브 페이지에서 테서랙트 설치가 가능하며, 자세한 설치 방법은 아래의 링크로 첨부하겠습니다.
- Tesseract 설치: [tesseract-ocr](https://github.com/tesseract-ocr/tessdoc)
- Tesseract 설치 방법: [2. Tesseract 부분 ~](https://velog.io/@wonjiny/EasyOCR%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%9C-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%B6%94%EC%B6%9C), [https://yunwoong.tistory.com/51](https://yunwoong.tistory.com/51)  

  
### Tesseract 실행하여 텍스트 수집해보기  
`tesseract imagename outputbase [-l lang] [--oem ocrenginemode] [--psm pagesegmode]`  

Tesseract 기본 실행 코드는 위와 같습니다.  
- `imagename`: OCR을 수행할 이미지 파일의 경로를 불러오면 됩니다.  
- `outputbase`: 출력 방식을 설정하는 것인데, 'stdout'이라고 작성할 경우 프롬프트 창에만 그 결과를 출력하고, 파일의 경로와 이름을 적어주면 출력 결과가 해당 경로와 이름에 맞게 파일로 저장됩니다.  
- `-l lang`: 어떤 언어로 인식할지 적어주는 부분입니다. 이 프로젝트에서는 한국어를 사용하므로 '-l kor'라고 적어주면 됩니다.  
- `--oem ocrenginemode`: OCR Engine Mode 설정인데, default로 주어진 값을 써도 무방합니다.  
- `--psm pagesegmode`: 페이지 분할 모드를 설정하는 것으로, 0~13까지의 값을 주어 바꿀 수 있으며, 자세한 사항은 cmd창에 'tesseract --help'를 쳐서 확인할 수 있습니다.  

📁 resources 폴더에 사용한 이미지와, 텍스트 추출결과 파일을 첨부하였습니다.(각각 test_text.png, res_text(2).txt)  

  
### 실제 텍스트 데이터와 OCR 인식 결과의 일치율  
과연 OCR 인식 결과를 NLP 감성분석 모델에 넘겨줘도 될만큼 정확한지를 확인하기 위해 실제 데이터와의 일치율을 확인해보았습니다.  
약 97%의 일치율을 보였으며, 해당 코드는 join_newlines_+_sequencematcher.ipynb 에서 확인가능합니다.  
감성분석 모델에 넘겨줘도 무방할 만큼 높은 정확도라고 판단되었습니다.  

  
### OCR 관련 앞으로의 과제  
1. 실시간으로 사용자 화면을 인식해오기  
➡️ 가장 중요한 부분 중 하나라고 할 수 있습니다. 스크린샷을 programatically 떠오는 방식을 선택할 예정인데요, 현재 여러 방법을 조사 중입니다. 이 중 사용자의 독서에 방해가 되지 않으면서, OCR을 통해 정확하게 스크린 샷을 얻을 수 있는 방법을 선정하고자 합니다.  
2. OCR로 인식해올 텍스트 양의 기준 정하기  
➡️ 텍스트 분석 결과를 통해 어떤 음악을 제공할지 결정이 되기 때문에, 그 양도 중요합니다. 따라서 회의와 여러 실험을 거쳐 텍스트 분석의 기준이 될 적정량을 정해야 할 것입니다.
3. Android studio와 연동 및 NLP 감성분석 모델과의 연결  
➡️ 추후 어플리케이션 개발 시 라이브러리 연동을 통해 안드로이드 스튜디오를 통해서도 연동을 할 예정입니다. 그 후, 인식 결과 추출된 텍스트 데이터를 NLP 감성분석 모델에 넘겨주는 데 초점을 둘 것입니다.
