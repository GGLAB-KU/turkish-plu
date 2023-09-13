# Summarization

This folder contains information regarding the summarization task.

### Requirements

Implemented with Python 3.7, using the following libraries:

```
transformers==4.27.0
evaluate==0.4.0
accelerate==0.15.0
torch==1.13.1
```

### Data

Find the train, validation, and test splits in the ```data``` folder, or at [Hugging Face](https://huggingface.co/datasets/ardauzunoglu/tr-wikihow-summ).

Summarization data follows the format below:

```
{
      "Şişeden Sinek Kapanı Yapmak": {
         "summary": "Boş bir meşrubat şişesi al. Şişenin üst kısmını kes. Kestiğin parçayı ters çevir. Şişenin iki parçasının kesik kenarlarını birleştir. Erimiş şeker karışımı hazırla. Sıvıyı bir kaşıkla şişenin huni ağzına dök. Başka bir tür yem kullan. Sirke ekle. Şişeyi güneş alan bir yere koy. Şişenin içine sık sık nefesini ver. Şişeyi at.",
         "document": "Kullanılmış bir şişe olabileceği gibi, meşrubatı döküp şişeyi de kullanabilirsin. Yeter ki içinde hiç meşrubat kalmasın ve şişe ılık suyla yıkanmış, kapağı çıkarılmış olsun. X Kaynağı araştır Makas kullan. Makasın sivri uçlarından birini kullanarak şişeye bir delik aç. Bunu, şişenin huni kısmının bittiği, geniş gövde kısmının başladığı yere yap (şişenin ortasına yakın bir yere). Şişeye delik açtıktan sonra makası sok ve şişeyi boydan boya kes. Şişenin huniye benzeyen üst kısmını tamamen ayır ki elinde iki parça olsun: huni (üst) ve sütun (alt). Huninin uç kısmına olabildiğince yakın kesmeye çalış yoksa ters çevirdiğinde yerinde durmaz. Şişenin huni kısmını kesmek için keskin bir bıçak da kullanabilirsin ama kendini kesmemeye özen göster. Bu projeyi çocuklarınla birlikte yapıyorsan en iyisi güvenli bir makas kullanman. Sonra da şişenin alt yarısının içine sok. Eğer huni kısmının ucuna yakın kestiysen, ters çevirip oturttuğunda parça orada duracaktır. Bunu yapmanın en basit ve en etkili yolu, kenarları zımbalamak. Şişenin çevresini birbirinden eşit uzaklıkta olacak biçimde üç ya da dört kez zımbala. X Kaynağı araştır Proje çocuklarla birlikte yapılıyorsa parçaları birbirine bir yetişkin zımbalamalı. Zımban yoksa aşağıdaki iki seçenek de iş görür. Bantlamak da iyi bir seçenektir ama bandın suya dayanıklı olması gerekiyor. Huninin etrafını üç ya da dört parça bantla birleştir. Eğer normal yapıştırıcı ya da Japon yapıştırıcısı kullanmak istiyorsan bunun da suya dayanıklı olması gerekiyor. Huniyi eklemeden önce, şişenin alt kısmının iç çeperine ince bir katman hâlinde yapıştırıcı sür. Sonra huni kısmını baş aşağı olacak biçimde şişenin içine oturt. Parmaklarını kullanarak huni ve alt kısmı birleştir. Yapıştırıcı kuruyana kadar iki parçayı yerli yerinde tut. Beş çorba kaşığı şekeri tencereye koy. Tencereyi ocağa yerleştir. Şekeri çıkıntı yapmayacak biçimde tencerenin dibine yay. X Kaynağı araştır Şekerin üstünü örtecek kadar su dök. Karışımı orta-yüksek sıcaklıkta ağır ağır ısıtarak kaynat. Karışımı iyice karıştır. Şekeri sıcak/ılık musluk suyunda eritirsen şekerli su elde edersin ama kaynatırsan sinekler için daha çekici olan konsantre bir \"şurup\" yapmış olursun. Sıvıyı ılık hâle gelene kadar beklet. Sıvının huninin kenarlarını kaplamasına izin ver ki huninin içine giren sinekler en başından yapışıp kalsın. Birkaç parça elma dilimle ve bunları huninin ağzından iterek şişenin içine at. Bir parça çiğ et ya da birkaç çorba kaşığı ekşimiş şarap da kullanabilirsin. Yalnızca şekerle ya da balla karışık su kullansan da olur. Eğer sıvı bir yem seçtiysen içine birkaç tatlı kaşığı sirke, tercihen de beyaz sirke koy. Bu, arıları ve yakalamak istemediğin diğer böcekleri kapandan uzak tutar. X Kaynağı araştır Bu, şişedeki meyvenin/etin çürümesini hızlandıracak ve sineklerin yemin kokusunu alma olasılığını artıracaktır. Aynı zamanda sıvı karışımın buharlaşmasını, böylece sinekleri kapana çeken bir feromon oluşturmasını da sağlayacaktır. Yeni sinek kapanının sinekleri nasıl yakaladığına hayret edeceksin. Bu, kapanın etkisini artırır çünkü böcekler sıcağa ve karbon dioksite gelir. Şişeyi ellerinin arasına alıp ovalayarak da ısıtabilirsin. Böceklerin birikmeye başladığını gördükten sonra şişeyi at ve yenisini yap. Er ya da geç yem etkisini yitirecek ve baştan başlaman gerekecek. Eğer şişeyi boşaltmaya çalışırsan bu çok zor olacaktır çünkü sinekler ve yem huninin içine yapışacaktır. Ayrıca ölü sinekleri eline alman hiç de iyi olmaz.",
         "article_title": "Sinek Tuzağı Yapmak",
         "article_url": "https://www.wikihow.com.tr/Sinek-Kapan%C4%B1-Nas%C4%B1l-Yap%C4%B1l%C4%B1r"
      },
      "Konserve Kutusundan Sinek Kapanı Yapmak": {
         "summary": "Uygun bir konserve kutusu bul. Birkaç parça tamir bandı kes. Bant şeritlerini konserve kutusunun etrafına sar. Tamir bandını konserve kutusundan sök. Konserve kutusunun iç kapağına küçük bir cep feneri bantla. Geceleri konserve kutusunu dışarı bırak. Sinekleri bekle. Kutuyu yenile.",
         "document": "Standart büyüklükte bir köpek maması ya da çorba konservesi bu iş için biçilmiş kaftandır. Kâğıt etiketi sök, konservenin kapağını çıkar, sonra kutuyu ılık suyla yıka. Kuruladıktan sonra sıradaki adıma geç. Bu parçaların kutunun etrafını saracak kadar uzun olması gerekiyor. Yapışkan uçlarına dokunmamaya ya da bunları kirletmemeye özen göster yoksa kapan işe yaramayacaktır. Yapışkanı kutuya aktarmak için tamir bandını hafifçe ovala. X Kaynağı araştır Kutunun yüzeyi artık yapış yapış olacaktır. Hafifçe değerek ne kadar yapışkan olduğuna bak. Çok yapışkan değilse yeni tamir bandı şeritleriyle tekrar dene. Kapak, cep fenerinin arkasına gelmeli. Böylece sinek kapanının alt kısmını yapmış olacaksın. Morötesi bir fener bulursan daha da iyi olur çünkü morötesi ışın çoğu sineği cezbeder. X Kaynağı araştır Kutu dik dursun ki yapışkan kısmının tümüyle sinekleri yakalayabilsin. Cep fenerini açıp kutunun içine oturt. Cep fenerinin de dik durduğundan ve pillerinin yeni olduğundan emin ol. X Kaynağı araştır Işığa gelecek ama kutunun yapışkan kenarlarına yapışacaklar. Eğer kutuyla başarılı biçimde sinek yakaladıysan en iyisi kutuyu atmak. Sineklere değmek zorunda kalmamak için, en iyisi kutuya eldivenle dokunmak. Kutuyu çöpe atmadan önce içine koymak için bir naylon torba hazırlaman iyi olabilir. X Kaynağı araştır",
         "article_title": "Sinek Tuzağı Yapmak",
         "article_url": "https://www.wikihow.com.tr/Sinek-Kapan%C4%B1-Nas%C4%B1l-Yap%C4%B1l%C4%B1r"
      },
      "Plastik ya da Cam Kaptan Sinek Kapanı Yapmak": {
         "summary": "Küçük bir kap al. Kaba sirke ekle. Sirkeye bulaşık deterjanı ekle. Meyve ya da çiğ et ekle. Kabı streç filmle kapla. Streç filmde delikler aç. Kabı dışarı koy. Sinekleri öldür. Ölü sinekleri at. Kabı dezenfekte et.",
         "document": "Bu cam bir kavanoz (mesela reçel kavanozu) ya da içine kuruyemiş veya fıstık ezmesi koyduğun cinsten plastik bir kap olabilir. Eğer kabın ya da kavanozun kapağı varsa çıkar. Bir şişe elma sirkesi al ve kabın içine yaklaşık 2,5 cm yükseklikte olacak kadar dök. Bu, sinekleri kaba çekecektir. X Kaynağı araştır Sirkenin yüzey gerilimini bozmak için, içine birkaç damla sıvı sabun ya da bulaşık deterjanı damlat. Yoksa sinekler sirkenin üstünde durup sirkeyi içebilirler. X Kaynağı araştır Kabın içine sirke/bulaşık deterjanı karışımı yerine meyve ve et de koyabilirsin. Sadece eklemek istediğin şeyi parçalara ayır ve kabın dibine koy. Çürüyen gıda kokusu sinekleri kaba çekecektir. X Kaynağı araştır En az 8 x 8 cm boyutlarında bir parça film kopar. Streç filmi ellerinle kabın ağzına güzelce ger. Eğer film yerinde durmuyorsa etrafına birkaç parça bant yapıştırabilir ya da bir lastik bant geçirebilirsin. Bıçak, makas, kürdan vb. kullanarak streç filmde en az dört delik aç. Bu, sineklerin kapanın içine girmesine olanak tanıyacak. X Kaynağı araştır Sinekler deliklerden kapanın içine girecektir. Ancak girdikleri delikleri tekrar bulamayacaklarından dışarı çıkmaları olanaksızdır. Aynı zamanda sinekler kabın içine koyduğun şeyi yemekle meşgul olacaktır. Sineklerin bir kısmı büyük olasılıkla kapanın içinde bir süre sonra ölecektir. Ama bazı sinekler hâlâ kabın içine koyduğun yemi yiyor olabilir. Kapanı eve al, lavabonun içine koy. Lavabo tıkacını takıp evyeyi sıcak suyla doldur. Evye dolunca kabı içine koyup on dakika bekle. Bu, sinekleri boğacaktır. Streç filmi sök ve at. Kabı çöp tenekesine götür ve kenarına vura vura boşalt. Bunu kapta hiç karışım ve sinek kalmayana kadar sürdür. X Kaynağı araştır Bunu kabı ılık su ve sabunla yıkayarak yapabilirsin. Kabın temizlenmesi ve tekrar kullanılabilmesi için, zararsız bazı kimyasal maddeler de kullanabilirsin. Kap temizlenince, tekrar kapan yapmak için kullanabilirsin.",
         "article_title": "Sinek Tuzağı Yapmak",
         "article_url": "https://www.wikihow.com.tr/Sinek-Kapan%C4%B1-Nas%C4%B1l-Yap%C4%B1l%C4%B1r"
      },
      "Kendi Yapışkan Sinek K.C3.A2ğıdını Yapmak": {
         "summary": "Kâğıt bir alışveriş poşeti al. Kâğıttan şeritler kes. Şeritlere delik aç. Delikten bir ip geçirip bağla. Şekerli karışım hazırla. Kâğıdı karışıma batır. Şeritleri as. Kâğıdı çöpe at.",
         "document": "Uzun ve yapışkan sinek şeritleri yapacağın için, poşetin uzun olması iyi olur. Naylon poşet kullanma çünkü yapışkan karışım naylona yapışmayacaktır.. Bir makas kullan ve 2,5 x 15 cm uzunlukta şeritler kes. Bunlardan dört ya da beş tanesine ihtiyacın olacak. Kestikten sonra şeritleri masanın üstüne koy. X Kaynağı araştır Makas ya da bıçak kullanarak şeritlerin ucuna 2,5 cm mesafede bir delik aç. Bunu her şerit için tekrarla. Elinin altında delgeç varsa şeritleri bununla da delebilirsin. X Kaynağı araştır En az on beş santim uzunlukta bir ip/tel kes. Her şerit için ipe ihtiyacın olacak. İpi şeridin deliğinden geçir ve düğüm at. Bir ölçek şeker, bir ölçek bal ve bir ölçek suyu bir tencereye koy. Tencereyi ocağa yerleştir ve orta-yüksek sıcaklıkta, malzemeler iyice karışana kadar ısıt. Hepsi karıştıktan sonra, karışımı oda sıcaklığına inmesi için beklet. X Kaynağı araştır Her bir şeridi karışıma batırarak şurupla kapla. Şeritleri bir pişirme kâğıdının üstüne koyup kurumaya bırak. Bir çivi ya da raptiye bulup şeritleri iplerinden as. Hepsini birbirine yakın yere koyabilir ya da evin farklı yerlerine yerleştirebilirsin. Şeritler yan yana durursa kapan daha etkili olur. X Kaynağı araştır Şeritler sinekle kaplanınca indirip çöpe at. Eğer herhangi bir sebepten ötürü, şeritler işe yaramadıysa muhtemelen üzerinde yeterince şurup olmadığı içindir. Yeniden şurup pişirip kâğıtları tekrar içine batır ya da sıfırdan başlayıp yeni şeritler yap.",
         "article_title": "Sinek Tuzağı Yapmak",
         "article_url": "https://www.wikihow.com.tr/Sinek-Kapan%C4%B1-Nas%C4%B1l-Yap%C4%B1l%C4%B1r"
      }
   }
```

### Training and Testing

Scripts for training and testing models and evaluating OOD models are provided.

For training, Hugging Face's Accelerate library is used. Therefore, run the following command for training:

```
accelerate config

accelerate test

accelerate launch train.py \
    --model_name_or_path MODEL_NAME \
    --dataset_name ardauzunoglu/tr-wikihow-summ \
    --num_train_epochs 3 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --num_beams 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 0.0001 \
    --hub_token HUB_TOKEN \
    --push_to_hub \
    --output_dir OUTPUT_MODEL_NAME
```

For testing, simply run the following command:

```
python3 test_summarization.py \
  --model_name_or_path MODEL_NAME \
  --per_device_test_batch_size 8 \
  --dataset_name ardauzunoglu/tr-wikihow-summ \
```

For evaluating OOD models, simply run the following command:

```
NUM_GPU=2
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=2

python3 -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID evaluate_baselines.py \
    --model_name MODEL_NAME \
```
