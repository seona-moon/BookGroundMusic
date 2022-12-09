
# 1. 음성데이터 확인 (eda)
## 1.1 데이터 구하기
http://www2.projects.science.uu.nl/memotion/emotifydata/

www2.projects.science.uu.nl


위 사이트에서 제공하는 4개의 음악장르(classical, pop, electronic, rock)별 각 100개의 음성 데이터파일을 사용했다.

우리의 타겟인 감성은 9가지로 분류되며(다중감성분류), 데이터 분포도를 확인해본 결과, 불균형을 보였다.

 

## 1.2. 데이터 불균형 해결
### 1.2.1. 유사한 특성끼리 합치기
 
데이터 불균형을 해결하기 위한 첫번째 시도는 유사한 특성끼리 묶는 것이다.
 

기쁨> amazement, joyful_activation
슬픔> sadness, nostalgia
중립> calmness, tenderness
분노> solemnity, power, tension

 

### 1.2.2. data augmentation
데이터 개수가 부족하기 때문에 undersampling이 아닌, oversampling을 수행해야 한다.

'sadness'에 해당하는 음성데이터에 관해 augmentation을 진행한다. 10개의 데이터만 추가하기로 결정했다.

(이에 대한 근거로는, 125:97=106:82 이기 때문에,...그냥 저자의 생각에 10개가 적당할 것 같았다.)

진행순서는 아래와 같다.

 
1. audiomentations 모듈을 설치해준다.
 
2. sadness에 해당하는 데이터를 불러온다.

3. sad_data 중에서 10개만 random하게 선택한다

4. augment 파라미터를 정해주고, augmented data를 얻는다.

5. 결과


 
# 2. 음성데이터 가공
## 2.1. train, validation dataset 
 

음악 장르별 균등하게 학습될 수 있게 하기 위해 다음의 순서에 따라 진행했다.

1. 음악 장르별 데이터프레임 만들기
2. 1.에서 제작한 데이터프레임을 train_test_split(test_size=0.2, random_state=0) 사용하여 분리
3. train df, validation df끼리 합치기 (pd.concat)
 
 
## 2.2. 음성 데이터 불러오기

음악 데이터를 불러오기 위해, librosa.load를 사용했다.
 파일 경로와 sample_rate를 입력받으면, signal ( type: array) 을 반환하는 함수를 만들었다. 



## 2.3. feature, label을 통해, train_data, train_label / val_data, val_label 만들기


앞서 정의한 load_audiofiles를 바탕으로 멜로디 특성을 담기 위한 함수를 정의했다.
타켓은 역시나 '감성, feeling, emotion' 이다.
tag를 사용하여 train데이터에만 augmeted data를 추가한다.

shape을 확인했을 때, 320+10=330으로 제대로 추가됨을 확인했다.

 

 
## 2.4. label encoding

target label이 str(문자열)이므로 LabelEncoder를 통해 int(정수)로 변환시킨다.

 
## 2.5. 음성 신호 전처리

( https://ahnjg.tistory.com/93 링크를 참고하였다.)

음성 신호를 전처리하기 위해, 성능이 일반적으로 많이 사용하는,
fourier transform + mel-filter = melspectrogram 을 사용했다.
사람의 청각기관을 반영했다는 특징역시 선택이유 중 하나이다.
melspectrogram을 통해 음성데이터를 변환해준다.

 


## 2.6. 정규화

mel_train, me_val 을 standardscaler로 정규화하고, (개수, 1, height, width)의 크기를 얻었다.



# 3. 모델 설계 및 평가
## 3.0. 모델 설계 
train 데이터 개수가 적기 때문에 과적합 가능성이 크다. 따라서 두가지 방법을 시도했다.

1. dropout

2. 적은 layer 

 
## 3.1. 합성곱층 4개, 합성곱 끝날때마다 dropout



cnn layer (4 layer, convolution) 를 직접 쌓았다. 

음성데이터이기 때문에 층을 너무 작지 않거나 크기않게 4개로 설정했다.

relu 활성화함수를 사용했고, 마지막에는 softmax를 사용했다.

각 합성곱이 끝날때마다 dropout, batchnormalization을 했다.

또한 4개의 다중감성 분류이므로, num_emotions=4로 설정했다. 

SGD optimizer를 사용했으며, learning rate=0.01, momentum=0.9로 설정했다.

loss는 crossentropyloss를 사용했다.

 
accuracy_score을 확인했을 때, 'y_true == y_pred'이라는 문구가 출력되었다. validation data에 관해 정확한 예측을 수행하였음을 확인했다.

loss 그래프를 그려본 결과, epoch가 진행함에도 변화하지 않았다.

inverse_transform을 사용하여 label encdoing 역변환하였을때, validiation data 예측결과를확인했다

 
 

## 3.2. 합성곱층 4개, 합성곱 끝날때마다 dropout + 파라미터 변화

SGD 대신 ADAM optimizer를 사용했고, learning rate=0.0001로 설정했다. 이외의 파라미터는 동일하다.

이번에도 loss가 상수함수의 꼴을 나타냈다. (즉, y_pred==y)


 
## 3.3. 결론
첫번째 모델을 최종모델로 선정하였다.

 

## 3.4. 데이터 구하기 
https://www.sellbuymusic.com/search/freebgm
https://pixabay.com/music/


음악 태그를 바탕으로 label을 붙어서 데이터를 수집중에 있다.

 

 

 

# 4. 음성 fade in, out 기술
## 4.1. fade out


sample audio data (.wav)를 다운받아서 진행했다.

저자가 사용한 파일은 기차의 휘슬소리로, 크게 2번 경적을 울린다.

fade out 함수를 만들어준다.

멜로디(주파수), 음압에 대해 각각 fade 효과를 적용해야 한다.

따라서, 입력받은 2차원 데이터를 각각 1차원 데이터로 처리하여(인덱싱) 선형연산을 해준다.

시간이 지날수록 소리가 작아져야 하기때문에, 점진적으로 작아지는 가중치를 곱해줘야 한다.

이를 위해 np.linspace(1.0,2.0, length)를 사용했다. 

4.3. 그래프 그리기에 정의한 함수를 바탕으로 결과를 출력해본 결과이다.

사진과 같이 fade-out이 잘 이루어진 것을 확인할 수 있다.

audio를 출력하여 직접 들어본 결과 역시 fade-out이 잘 이뤄짐을 확인할 수 있었다.

 

 


## 4.2. fade in
 
4.1. fade out 과 유사하게 진행하되, 연산위치와 가중치만 바꿔주면 된다.

연산위치는 음악의 앞부분, 

가중치는 점진적으로 커쳐야하기 때문에, linspace(0.0,1.0,length)를 사용한다.

결과는 위와 같이 예쁘게 fade in이 된 것을 확인 할 수 있다.

 

 

## 4.3. 그래프 그리기 


새롭게 정의한 함수는 위의 두 가지이다.

하나는 waveform(소리의 파동) 출력함수이고, 하나는 specgram(소리의 스펙트럼) 출력함수이다.

 

파형에서는 시간축의 변화에 따른 진폭 축의 변화를 볼 수 있고, 스펙트럼에서는 주파수 축의 변화에 따른 진폭 축의 변화를 볼 수 있는 반면, 스펙트로그램에서는 시간축과 주파수 축의 변화에 따라 진폭의 차이를 인쇄 농도 / 표시 색상의 차이로 나타낸다. (http://ko.wordow.com/english/dictionary/spectrogram)
 
 

 

 

# 5. 추후계획
- 감성이 라벨링된 크기가 큰 데이터셋 찾기
- 모델 성능 개선
- 사용자 취향에 따른, 음악 장르 설정 및 음악 강도 조절 기술 구현하기
 

 

# 6. 참고문헌
https://stackoverflow.com/questions/64894809/is-there-a-way-to-make-fade-out-by-librosa-or-another-on-python
https://www.kaggle.com/code/sjmin99/w11p2-solution-code
https://ahnjg.tistory.com/93 
http://ko.wordow.com/english/dictionary/spectrogram
