
# Global Ai Hub:Kırmızı Şarap Veri Seti Sınıflandırması

Global AI Hub Makine öğrenmesi Bootcampinde katılımcılardan  sınıflandırma yada regresyon projeleri yapmaları istenmiştir.

## 1. Proje Konusu ve Veri Seti

Bu projede proje mentörlerinin önermiş olduğu Wine quality dataseti üzerinden  makine öğrenmesinde sınıflandırma konusu seçilmiştir.

## 2. EDA
Veri Seti Hakkında:

- Veri Seti 1599 satır 12 sütundan oluşmaktadır.
- Verindeki kolonlar:'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol', 'quality'.
- 'quality' kolonu hedef(bağımlı) değişkendir ve değişkenin veri türü integerdır. Bağımsız değişkenlerin türü ise float tur.
![image](https://github.com/cagandemirmr/Wine_Quality_Clustering/assets/145290776/ac4a4ef5-b798-4727-9978-949b0f972ed9)

- Kolonlardaki çeşitilik ise:

![image](https://github.com/cagandemirmr/Wine_Quality_Clustering/assets/145290776/9a6d23d4-8b43-46f2-b8be-130f89148e6d)


- Veri setinde boş bir değer bulunmamaktadır.
- Ancak veri setinde bir çok aykırı değer bulunmaktadır.Bu değerlerin şarabın kalitesinde etkinin olduğunu düşündüğüm için ve veri çeşitliliğini arttırdığı için aykırı değerleri tutmaya karar verdim.

![image](https://github.com/cagandemirmr/Wine_Quality_Clustering/assets/145290776/38b287dd-3026-4731-ba2c-66aad285ebea)

- Kaliteyi etkileyen bağımsız değişkenlerin ilişkisini incelemek için korelasyona, korelasyonda metoda karar vermem için ise verilerin dağılımını test etmem gerekliydi.Bu nedenle öncelikle veri setimin küçük olduğunu düşünerek shapiro-wilks testi yaparım,eğer p değeri 0.05 ten küçük çıkarsa dağılımın normal olduğunu reddedemezdim. Ancak shapiro-wilks testinin doğru olmama olasılığına karşılık jarque-bera testini uyguladım.P değeri bu testte de her bağımsız değişken sonucunda veri dağılımının normal olduğunu reddedemezdim. Veri değılımının normal olduğunu ispat ettiğim için korelasyonda 'Spearman' metodunu kullandım. Korelasyon sonucunda, yüksek korelasyonda(+-0.6 ve üstü) olan değişkenlere rastlamadım.
- Aykırı değerleri kontrol ederken birden fazla bağımsız değer üzerinden aykırı değer olup olmadığını değerlendirebilmek için LocalOutlierFactor kullanırım. Ancak Aykırı değerlerin bir çoğunun yüksek kalitede olmasından dolayı aykırı değerlerin kalmasına karar verdim.Verileri silmemdeki bir diğer neden ise veri setinden bir kaç veri silindiğimde seçtiğim modellerde belirgin bir performans düşüşü gözlemlememdi.

![image](https://github.com/cagandemirmr/Wine_Quality_Clustering/assets/145290776/ac94759e-8da3-4f2a-a505-6da29425efab)


Seçmiş olduğum sınıflandırma modelleri :

                                        Logistic Regression(LG)
                                        Ridge Regression(RR)
                                        Decision Tree(DT)
                                        Naive Bayes(NB)
                                        Neural Network(NN)

Veri silmeden önce gözlemlediğim accuracy oranları:

LG:0.61
RR:0.58
DT:0.56
NB:0.56
NN:0.57

Duplicate veriler silindikten sonraki accuracy oranları:

LG:0.58
RR:0.57
DT:0.50
NB:0.50
NN:0.59

Çok düşük korelasyonda olan verileri sildiğimde accuracy oranları:

LG:0.55
RR:0.55
DT:0.49
NB:0.45
NN:0.56


## 3.Veri Önişleme

-Feature Engineering ile ilgili olarak yeni feature lar ekledim.Bu featureları eklerken kaliteye etkili olan +-0.30-0.40 korelasyonların, kendi içerisinde +-0,30-0,40 korelasyonlarında olan özellikler arasında bölmeyle ve çarpmayla olucak şekilde iki ayrı deneme yaptım.

'''Ekleyeceğim featurelar;
citric acid/fixed acidity,
citric acid/ph,
citric acid/volatile acidity,
density/fixed acidity,
fixed acidity/ph,
free sulfur dioxide/total sulfur dioxide'''

Çarpma işlemiyle oluşturduğum featurelar:

'''df2.loc[:,"ca/fa"] = df2["citric acid"] * df2["fixed acidity"]
df2.loc[:,"ca/ph"] = df2["citric acid"] * df2["pH"]
df2.loc[:,"d/fa"] = df2["density"] * df2["fixed acidity"]
df2.loc[:,"fa/ph"] = df2["fixed acidity"] * df2["pH"]
df2.loc[:,"fsd/tsd"] = df2['free sulfur dioxide'] * df2["total sulfur dioxide"]
df2.loc[:,"va/ca"] = df2["volatile acidity"] * df2["citric acid"]
df2.loc[:,"va/s"] = df2["volatile acidity"] * df2["sulphates"]
df2.loc[:,"va/fa"] = df2["volatile acidity"] * df2["fixed acidity"]
df2.loc[:,"s/ca"] = df2["sulphates"] * df2["citric acid"]
df2.loc[:,"s/al"] = df2["sulphates"] * df2["alcohol"]
df2.loc[:,"s/fa"] = df2["sulphates"] * df2["fixed acidity"]
df2.loc[:,"a/cl"] = df2["alcohol"] * df2["chlorides"]
df2.loc[:,"a/d"] = df2["alcohol"] * df2["density"]
df2.loc[:,"a/ph"] = df2["alcohol"] * df2["sulphates"]'''

Çarpma işlemiyle oluşturduğum featureların accuracy oranları:

LG:0.51
RR:0.54
DT:0.45
NB:0.45
NN:0.49


Bölmeyle oluşturduğum Featurelar:

df2.loc[:,"ca/fa"] = df2["citric acid"] / df2["fixed acidity"]
df2.loc[:,"ca/ph"] = df2["citric acid"] / df2["pH"]
df2.loc[:,"d/fa"] = df2["density"] / df2["fixed acidity"]
df2.loc[:,"fa/ph"] = df2["fixed acidity"] / df2["pH"]
df2.loc[:,"fsd/tsd"] = df2['free sulfur dioxide'] / df2["total sulfur dioxide"]
df2.loc[:,"va/ca"] = df2["volatile acidity"] / df2["citric acid"]
df2.loc[:,"va/s"] = df2["volatile acidity"] / df2["sulphates"]
df2.loc[:,"va/fa"] = df2["volatile acidity"] / df2["fixed acidity"]
df2.loc[:,"s/ca"] = df2["sulphates"] / df2["citric acid"]
df2.loc[:,"s/al"] = df2["sulphates"] / df2["alcohol"]
df2.loc[:,"s/fa"] = df2["sulphates"] / df2["fixed acidity"]
df2.loc[:,"a/cl"] = df2["alcohol"] / df2["chlorides"]
df2.loc[:,"a/d"] = df2["alcohol"] / df2["density"]
df2.loc[:,"a/ph"] = df2["alcohol"] / df2["sulphates"]

Bölme işlemiyle oluşturduğum featureların accuracy oranları:

LG:0.51
RR:0.54
DT:0.45
NB:0.45
NN:0.49

- İlginç bir şekilde yeni feature işlemlerini modeller üzerinde sonuçları aldığımda en yüksek değerin yeni feature üretilmemiş değerler olduğunu gördüm. İkinci sırada bölme işleminde yüksek değerleri gözlemlerken en kötü değerleri çarpma işlemi vermiştir.

- Bağımsız değişkenlerde ise performans olarak MinMaxScalerın modellerde en iyi değeri göstermesi üzerine bu scaler'ı uyguladım.

MinMaxScaler Uygulandığında accuracy değerleri:

LG:0.60
RR:0.59
DT:0.48
NB:0.48
NN:0.61

RobustScaler Uygulandığında accuracy değerleri:

LG:0.61
RR:0.58
DT:0.49
NB:0.49
NN:0.56


- Kalite hedef değişkeni ise 6 dan düşük olan değerlere "kötü" 6 dan yüksek değerler "iyi" olacak şekilde kategorik değişken atarım.Ve "kotu" ve iyi değerlere LabelEncoder uygulayarak "kotu" değerlere 0, "iyi" olarak belirttiğim kategorik değişkenlere 1 değerine atarım.

Label Encoding Uygulandığında accuracy değerleri:

LG:0.77
RR:0.76
DT:0.71
NB:0.71
NN:0.79



- Accuracy score un düşük olmasından dolayı Chatgptye 1134 satırlık sentetik bir veri üreterek modelin performansını arttırmaya çalıştım.Bu nedenle sentetik veriyi mevcut verilerimle birleştirerek modelime entegre ettim.Bunu yaparken veri eksilttiğimde modellerin performansındaki düşüşü gözlemlememdi.
![image](https://github.com/cagandemirmr/Wine_Quality_Clustering/assets/145290776/4265dce6-41e8-45d9-bc16-6dea748dbe78)

Sonuçlar:

LG:0.73
RR:0.73
DT:0.83
NB:0.83
NN:0.76

## Model Seçimi
Bir kaç ayarlamadan sonra modelleri eğiterek sonuçlar aldım.

![image](https://github.com/cagandemirmr/Wine_Quality_Clustering/assets/145290776/d4cf272b-35f9-4319-944e-5d8902b852c6)

![image](https://github.com/cagandemirmr/Wine_Quality_Clustering/assets/145290776/d04794d9-2543-42a2-a3ea-b6ebddb2b31a)



Bu sonuçlara göre en iyi seçimin Decision Tree olduğuna karar verdim ve Cross Validation işlemi yaptım.

Ancak Cross Validation işlemi sonucunda CV'i 500 kere uygulayarak en iyi sonuç olan 0.85 sonucunu buldum.

## HiperParametre Analizi

Grid Search Cv kullanarak 
'criterion','splitter','max_depth','min_samples_split','min_samples_leaf','max_features','max_leaf_nodes','random_state' parametrelerineilgili parametrelerin tamamını belirledim.

![image](https://github.com/cagandemirmr/Wine_Quality_Clustering/assets/145290776/045760e9-63e8-46cf-add1-0e8c9ea479af)


![image](https://github.com/cagandemirmr/Wine_Quality_Clustering/assets/145290776/be0e6a67-c68c-4149-904e-ac45552888ea)

Bunun sonucunda;criterion="gini",splitter="best",max_depth=20,min_samples_split=2 sonucunu aldım ve buna göre modelimi test ettim.Ve sonuç olarak 0.85 accuracy rateini aldım
![image](https://github.com/cagandemirmr/Wine_Quality_Clustering/assets/145290776/67b39f61-4114-4b6a-8430-bf3c21375872)


## Veri Setinin Tekrar  Değerlendirilmesi

Sentetik Veri Setinin ChatGPT ile üretilmesi yerine imbalanced kütüphanesinden SMOTE ile KNN modeliyle eğitilmesinin daha doğru olacağını düşündüm.

![image](https://github.com/cagandemirmr/Wine_Quality_Clustering/assets/145290776/48258ec2-e42e-464d-a1c4-1446a9c2aad1)


Bunun üzerine Yapay veriyi doğal veriyle karıştırarak ve bunu yaparken KNN mantığını kullanan SMOTE ve azınlık verinin arttırılmasında yardımcı olan ADASYN modellerini kullanarak verimi çoğalttım.

Bu işlemi yaptıktan sonra ciddi anlamda korelasyon seviyesi yüksek verilerle karşılaştım(+-0.60-0.90).
![image](https://github.com/cagandemirmr/Wine_Quality_Clustering/assets/145290776/729e1ee1-f2df-4cf5-9ed3-ba54992fd7c6)


-Veri yapısını düzeltmek için ancak modelin performansını olumsuz etkilememek için veri setinde en az etkili olan değerin aykırı değerlerini çıkarttım.

![image](https://github.com/cagandemirmr/Wine_Quality_Clustering/assets/145290776/2fbeb729-4c88-4fe3-afe2-2c879759834d)

MinMaxScaler ve LabelEncoding işlemini yaptıktan sonra modelimi eğittim ve 0.91 accuracy sonucunu aldım.

GridSearchCV de yaptığım hiperparametre analizi sonucunda en iyi değerlerin "criterion="gini",splitter="best",max_depth=5,min_samples_split=2,max_leaf_nodes=10" olduğunu gördüm.

![image](https://github.com/cagandemirmr/Wine_Quality_Clustering/assets/145290776/500dea9b-b0bb-4ab1-9c43-e91630a642ae)


Modelimi eğittiğimde ise accuracy score um 0.93 olmuştu.
![image](https://github.com/cagandemirmr/Wine_Quality_Clustering/assets/145290776/8a9bece3-490d-4bd7-8883-d7d7bf8e0252)






## Global Ai ve Mentörlerine Teşekkürler

-  Bu bootcampin olmasını mümkün kılan Global Ai ve Global Ai mentörlerine teşekkür ederim.
- Projenin ilk  aşamasında özellikle Göker Güner'e teşekkür ederim.
  
