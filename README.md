# TUBITAK-2209B

Pekiştirmeli öğrenme ile otonom it dalaşı kurgulamak.

## Modelleme ve Simülasyon

Otopilot tasarımı ve simülasyon için F-16 modeli gerekiyor. Eğer mümkün
gözüküyorsa hasar verme işlevselliğinin eklenmesi tartışılacak.

### Seyrüsefer sistemi

Yoksa seyrüsefer sisteminin eklenmesi gerekiyor.

### Sensörler

F-16 veya benzer sistemlerde hangi sensörlerin olduğu raporlanacak. Bu
sensörler aracılığı ile hangi bilgilere erişilebileceği araştırılacak. Sensör
verilerinin hangi aralıkta olduğu kestirilecek.

## Otopilot

Aynı model üzerinde otopilot algoritması da yazılacak. Olası kontrolcü tasarım
yöntemleri:
1. Robust control
2. Nonlinear dynamic inversion
3. Adaptive control
4. Thrust vector control

@aydinke16 enerji-manevra teorisine bakacak. One-step look ahead metodu da
araştırılacak. Danışmanlık için Barış Başpınar'a sorulacak sorular:
1. Enerji teorisi sektörde kullanılıyor mu?
2. Gelecek vaat eden bir yöntem mi yoksa geride kalmaya mı başladı?
3. Alternatif yöntemler nelerdir?

## Yapay Zeka Algoritması

### Aksiyon yapılandırması

#### Analitik fonksiyon

Parametreler kullanılarak yol haritası çizilmesi ve koordinatların otopilota
yedirilmesi ile düşünülmüş yol haritası. Olası matematiksel yöntemler
- Eigencurves
- Bezier Curves
- B-spline
