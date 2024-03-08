# NBART-Multilingual-Translator

**This code demonstrates how to use an NBART (Neural Bidirectional AutoRegressive Transformer) pre-trained multilingual model to perform translation tasks between different languages. The model is trained on multiple language pairs using data parallelism, which allows it to learn representations across all languages simultaneously. It has been fine-tuned on various datasets such as WMT20, TED Talks, and WikiMatrix. Here are some examples of translations performed by the model:**

 * From English to Arabic: "We can help you create a list of Page members with Vacation Rental Properties and relevant information like email addresses, websites, and links to profiles." → "نحن نساعدك في إنشاء قائمة بـPage مع عقارات إيجار الفنادق ومعلومات مرتبطة مثل عناوين البريد الإلكتروني ومواقع الويب وروابط إلى ملفاتهم".
 * From Arabic to English: "نحن نساعدك في إنشاء قائمة بـPage مع عقارات إيجار الفنادق ومعلومات مرتبطة مثل عناوين البريد الإلكتروني ومواقع الويب وروابط إلى ملفاتهم". → "We can help you create a list of Page members with Vacation Rental Properties and associated information like email addresses, websites, and links to profiles."
 * From English to German: "We can help you create a list of Page members with Vacation Rental Properties and associated information like email addresses, websites, and links to profiles." → "Wir können Ihnen helfen, eine Liste von Seitenmitgliedern mit Urlaubsvermietungsobjekten und verbundenen Informationen wie E-Mail-Adressen, Websites und Verknüpfungen zu Profilen zu erstellen."
 * From German to English: "Wir können Ihnen helfen, eine Liste von Seitenmitgliedern mit Urlaubsvermietungsobjekten und verbundenen Informationen wie E-Mail-Adressen, Websites und Verknüpfungen zu Profilen zu erstellen." → "We can help you create a list of Page members with vacation rental properties and associated information like email addresses, websites, and links to profiles."
 * From English to Spanish: "We can help you create a list of Page members with Vacation Rental Properties and associated information like email addresses, websites, and links to profiles." → "Podemos ayudarte a crear una lista de miembros de Página con propiedades de alquiler de vacaciones y información asociada como direcciones de correo electrónico, sitios web y vínculos a perfiles."
 * From Spanish to English: "Podemos ayudarte a crear una lista de miembros de Página con propiedades de alquiler de vacaciones y información asociada como direcciones de correo electrónico, sitios web y vínculos a perfiles." → "We can help you create a list of Page members with vacation rental properties and associated information like email addresses, websites, and links to profiles."
 * From English to Chinese: "We can help you create a list of Page members with Vacation Rental Properties and associated information like email addresses, websites, and links to profiles." → "我们可以帮助您创建一份具有度假出租房屋和相关信息（如电子邮件地址，网站和指向个人资料的链接）的Page成员列表。"
 * From Chinese to English: "我们可以帮助您创建一份具有度假出租房屋和相关信息（如电子邮件地址，网站和指向个人资料的链接）的Page成员列表。"→ "We can help you create a list of Page members with vacation rental properties and associated information like email addresses, websites, and links to profiles."

> Please note that machine translation is not always accurate or appropriate in every situation, so it may be necessary to review and edit translated text before using it.


```Python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
```
