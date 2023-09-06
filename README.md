# ⚖ Turkish Procedural Language Understanding Benchmark

Turkish PLU Benchmark is the first and single (so far) resource to study procedural instructions in Turkish. It presents a document-level umbrella corpus of how-to tutorials and six different task-specific datasets. The corpus is obtained by translating a portion of the [English wikiHow corpus](https://github.com/zharry29/wikihow-goal-step) using [Google Translate](https://cloud.google.com/translate) and scraping the [Turkish wikiHow](https://www.wikihow.com.tr), while the task-specific datasets are derived from the corpus itself. 

# 📖 Corpus

We utilize wikiHow, a large-scale source for procedural texts that contains how-to tutorials in a wide range of domains, from Pets and Animals to Crafts and Hobbies, to create our corpus. We only focus on the following categories: Cars and Other Vehicles, Computers and Electronics, Health, Hobbies and Crafts, Home and Garden, Pets and Animals. Unlike categories like Relationships or Youth, they contain well-defined, more objective articles that are better suited to evaluate procedural language understanding abilities. We follow the file format of [English wikiHow corpus](https://github.com/zharry29/wikihow-goal-step) and extract title, methods/parts, steps, as well as the additional information, such as the related tutorials, references, tips, and warnings. An example tutorial can be seen [here](https://github.com/ardauzunoglu/turkish-plu/blob/main/example_tutorial.json).

Our corpus creation method, corpus statistics, and machine translation quality control can be reviewed in our [paper](). You can download our corpus [here]().

# 🥅 Tasks

While the Turkish wikiHow corpus constitutes an umbrella dataset, we also create sub-groups of data for each procedural task we study.

### Linking Actions

**Definition:** Linking actions task aims to find the counterpart of a given action in another set of actions. <br>
**Example:** 
```
{
  "FAT32 olacak şekilde flash belleği biçimlendir.": "USB Sürücü Nasıl Biçimlendirilir"
}
```
### Goal Inference

**Definition:** Goal inference task aims to predict the step that helps achieving the given prompt goal, formulated as a multiple choice task. <br>
**Example:** 
```
{
  "Step": "İpeği nazikçe kendisine sürterek öğeyi yıkayın.",
  "Candidate0": "Porselen Temizlemek",
  "Candidate1": "Kadife Temizlemek",
  "Candidate2": "İpek Temizlemek",
  "Candidate3": "Kauçuk Temizlemek",
  "label":2
}
```
### Step Inference

**Definition:** Step inference task aims to predict the goal that the given prompt step helps achieving, formulated as a multiple choice task. <br>
**Example:** 
```
{
  "Goal": "Cam Pipo Temizlemek",
  "Candidate0": "Gideri kaynar suyla durula.",
  "Candidate1": "Ahşabı temiz suyla durula.",
  "Candidate2": "Tezgâhı taze suyla durula.",
  "Candidate3": "Pipoyu sıcak suyla durula.",
  "label":3
}
```
### Step Ordering

**Definition:** Step ordering task aims to predict the preceding step out of the two given steps that helps achieving a given goal, formulated as a multiple choice task. <br>
**Example:** 
```
{
  "Goal": "Bir Google Hesabını Engellemek",
  "Candidate0": "Kullanıcının profil adının yanında bulunan aşağı oka tıklayın.",
  "Candidate1": "Engellenmesini istediğiniz kullanıcının Google+ profiline gidin.",
  "label":1
}
```
### Next Event Prediction

**Definition:** Next event prediction task aims to produce the following action for a given context. Although it also can be formulated as a text generation task, we approach it as a multiple-choice task. <br>
**Example:** 
```
{
  "Goal": "Tuzlu Su Çözeltisi Yapmak",
  "Step": "1/2 tatlı kaşığı (2,5 g) tuzu bir bardağa koy."
  "Candidate0": "350 ml suyu bir tencereye koyup kaynat.",
  "Candidate1": "Karışıma 2 ya da 3 damla sıvı gıda boyası ekleyip tekrar karıştır.",
  "Candidate2": "İçine 1/2 çorba kaşığı karbonat ekleyip karıştır.",
  "Candidate3": "230 ml (1 bardak) sıcak suyu ekleyip iyice karıştır.",
  "label":3
}
```
### Summarization

**Definition:** Summarization task aims to derive the key takeaways from long texts. We choose abstractive summarization as our objective. <br>
**Example:** 
```
{
  "Bir Geri Dönüşüm Merkezine Karton Vermek": {
         "summary": "Kartonları kağıt, teneke kutu ve şişe gibi diğer geri dönüştürülebilir maddelerden ayırın. Geri dönüştürülemeyen ıslak veya kirlenmiş kartonları çıkarın. Kolay toplama veya taşıma için karton kutuları düzleştirin. Geri dönüşüm servisinizin evinizden karton alıp almadığını kontrol edin. Yerel bir geri dönüşüm tesisine karton bırakın.",
         "document": "Bazı konumlardaki geri dönüşüm programları, farklı türde geri dönüştürülebilir maddeleri toplama veya bırakma için ayırmanızı gerektirir. Diğer programlarda, tüm geri dönüşümlerin sizin için bir tesiste sıralandığı “tek akışlı geri dönüşüm” vardır. Karton kutu veya çarşafları, mısır gevreği kutularını, boş kağıt havlu veya tuvalet kağıdı rulolarını, kartonları ve ayakkabı kutularını geri dönüştürebilirsiniz. Geri dönüşüm sırasında filtrelenen karton kutuların üzerine koli bandı veya nakliye etiketleri bırakabilirsiniz. Islak kartondaki lifler sertleşir ve bu da geri dönüşüm sürecini etkileyebilir. Pizza kutuları gibi karton kutular, tesislerdeki diğer geri dönüştürülebilir maddeleri bozabilecek gres ve yağ içerir. Islak veya kirlenmiş kartonları çöp kutusuna atın. Paket servis kapları ve kahve fincanları gibi yiyecek izleri olan çoğu karton kaplar, geri dönüştürülemiyorsa atılmalı veya kompostlanmalıdır. Posta veya paketlenmiş mallar için kullanılan karton kutuları, depolanabilmeleri veya düz bir şekilde taşınabilmeleri için ayırın. Bir kutu kesici veya makasla birlikte herhangi bir bant tutan kanatları dikkatlice kesin. Kanatları ayırın ve düzleştirmek için kutuyu aşağı bastırın. Kaldırım kenarı geri dönüşüm veya bırakma tesisleri genellikle karton kutuların düz bir şekilde parçalanmasını gerektirir. Çoğu hizmet, karton kutuları ücretsiz olarak alacaktır. Teslim alma hizmetiniz yoksa, bölgenize hizmet veren yerel şirketler için web'de \"geri dönüşümle teslim alma\" veya \"konut geri dönüşümü\" araması yapın. Pennsylvania, New Jersey ve District of Columbia gibi bazı eyaletlerde, sakinlerin geri dönüşüm yapmasını gerektiren zorunlu yasalar vardır. Karton gibi geri dönüşüm malzemelerini çöpe atanlara para cezası kesilebilir. Evinizdeki geri dönüşüm kutularına sığmayacak çok miktarda kartonunuz varsa, kartonu yerel bir tesise götürebilirsiniz. Kartonunuzu alacak yakın bir yer bulmak için web'de \"geri dönüşüm kartonunu bırakın\" ifadesini arayın. Evinizden karton toplayan bir geri dönüşüm servisiniz varsa, genellikle sizin için geri dönüşüm kutuları sağlarlar. Bazı tesislerde, kartonu küçük kompakt balyalar halinde sıkıştıran bir karton balya makinesi bulunabilir. Yerel geri dönüşüm tesisinizde kullanıma uygun bir yerinde olabilir veya yalnızca çalışanlar için olabilir.",
         "article_title": "Karton Geri Dönüştürmek",
         "article_url": "https://www.wikihow.com/Recycle-Cardboard"
      }
}
```
# 🤖 Baseline Models

Goal Inference: [BERTurk](https://huggingface.co/ardauzunoglu/BERTurk-GI), [DistilBERTurk](https://huggingface.co/ardauzunoglu/DistilBERTurk-GI), [XLM-R](https://huggingface.co/ardauzunoglu/XLM-R-Turkish-GI) <br>
Step Ordering: [BERTurk](https://huggingface.co/ardauzunoglu/BERTurk-SO), [DistilBERTurk](https://huggingface.co/ardauzunoglu/DistilBERTurk-SO), [XLM-R](https://huggingface.co/ardauzunoglu/XLM-R-Turkish-SO) <br>
Next Event Prediction: [BERTurk](https://huggingface.co/ardauzunoglu/BERTurk-NEP), [DistilBERTurk](https://huggingface.co/ardauzunoglu/DistilBERTurk-NEP), [XLM-R](https://huggingface.co/ardauzunoglu/XLM-R-Turkish-NEP) <br>
Summarization: [mBART](https://huggingface.co/ardauzunoglu/mbart-pro-summ), [mT5](https://huggingface.co/ardauzunoglu/mt5-base-pro-summ), [TR-BART](https://huggingface.co/ardauzunoglu/tr-bart-pro-summ) <br>
SimCSE Models: [Supervised BERT](https://huggingface.co/ardauzunoglu/sup-simcse-tr-bert-base), [Unsupervised BERT](https://huggingface.co/ardauzunoglu/unsup-simcse-tr-bert-base), [Supervised XLM-R](https://huggingface.co/ardauzunoglu/sup-simcse-tr-xlm-roberta-base), [Unsupervised XLM-R](https://huggingface.co/ardauzunoglu/unsup-simcse-tr-xlm-roberta-base) <br>

# 🖊️ Citation

To be updated upon publication.

# ℹ️ Acknowledgement

To be updated upon publication.
