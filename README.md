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

## **Python Code**

```Python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
```

```Python
article_en = "Yes, we can do that. We can extract the information you need from the Facebook group page and compile a list of Page members with Vacation Rental Property and related information, including their emails, website, and link to their profile."

# translate Arabic to English
tokenizer.src_lang = "en_XX"
encoded_en = tokenizer(article_en, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_en,
    forced_bos_token_id=tokenizer.lang_code_to_id["ar_AR"]
)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

#Output: ['نعم، يمكننا أن نفعل ذلك. يمكننا استخراج المعلومات التي تحتاجها من صفحة مجموعة الفيسبوك وجمع قائمة من أعضاء الصفحة مع عقارات الفنادق والمعلومات المرتبطة، بما في ذلك بريدهم الإلكتروني، موقعهم، وروابطهم إلى ملفهم.']
```

```Python
article_en = 'نعم، يمكننا أن نفعل ذلك. يمكننا استخراج المعلومات التي تحتاجها من صفحة مجموعة الفيسبوك وجمع قائمة من أعضاء الصفحة مع عقارات الفنادق والمعلومات المرتبطة، بما في ذلك بريدهم الإلكتروني، موقعهم، وروابطهم إلى ملفهم.'

# translate English to Arabic
tokenizer.src_lang = "ar_AR"
encoded_en = tokenizer(article_en, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_en,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

#Output: ['Yes, we can do that. We can extract the information you need from a Facebook group page and pull together a list of the members of the page with their hotel properties and related information, including their email address, their website, and link them to their profile.']
```

```Python
article_en = "Yes, we can do that. We can extract the information you need from the Facebook group page and compile a list of Page members with Vacation Rental Property and related information, including their emails, website, and link to their profile."

# translate German to English
tokenizer.src_lang = "en_XX"
encoded_en = tokenizer(article_en, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_en,
    forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"]
)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

#Output: ['Ja, wir können das tun. Wir können die benötigten Informationen aus der Facebook-Gruppenseite extrahieren und eine Liste der Mitglieder der Seite mit Vacation Rental Property und verwandten Informationen zusammenfassen, einschließlich ihrer E-Mails, Website und Link zu ihrem Profil.']
```

```Python
article_en = 'Ja, wir können das tun. Wir können die benötigten Informationen aus der Facebook-Gruppenseite extrahieren und eine Liste der Mitglieder der Seite mit Vacation Rental Property und verwandten Informationen zusammenfassen, einschließlich ihrer E-Mails, Website und Link zu ihrem Profil.'

# translate English to German
tokenizer.src_lang = "de_DE"
encoded_en = tokenizer(article_en, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_en,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

#Output: ['Yes, we can do that. We can extract the required information from the Facebook group page and compile a list of the members of the Vacation Rental Property page and related information, including their emails, website and link to their profile.']
```

```Python
article_en = "Yes, we can do that. We can extract the information you need from the Facebook group page and compile a list of Page members with Vacation Rental Property and related information, including their emails, website, and link to their profile."

# translate Spanish to English
tokenizer.src_lang = "en_XX"
encoded_en = tokenizer(article_en, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_en,
    forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"]
)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

#Output: ['Sí, podemos extraer la información necesaria de la página de grupo de Facebook y compilar una lista de los miembros de la página con Propiedad Alquiladora de Vacaciones y información relacionada, incluyendo sus correos electrónicos, página web y enlace a su perfil.']
```

```Python
article_en = 'Sí, podemos extraer la información necesaria de la página de grupo de Facebook y compilar una lista de los miembros de la página con Propiedad Alquiladora de Vacaciones y información relacionada, incluyendo sus correos electrónicos, página web y enlace a su perfil..'

# translate English to Spanish
tokenizer.src_lang = "es_XX"
encoded_en = tokenizer(article_en, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_en,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

#Output: ['Yes, we can extract the necessary information from the Facebook group page and compile a list of the members of the Vacation Rental Properties page and related information, including their emails, website and link to their profile.']
```

```Python
article_en = "Yes, we can do that. We can extract the information you need from the Facebook group page and compile a list of Page members with Vacation Rental Property and related information, including their emails, website, and link to their profile."

# translate Chinese to English
tokenizer.src_lang = "en_XX"
encoded_en = tokenizer(article_en, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_en,
    forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

#Output: ['是的,我们可以做到这一点。我们可以从Facebook小组页面中提取你所需要的信息,并汇编一份与度假租用物业有关的会员名单,包括他们的电子邮件、网站,并链接到他们的个人资料。']
```

```Python
article_en = '是的,我们可以做到这一点。我们可以从Facebook小组页面中提取你所需要的信息,并汇编一份与度假租用物业有关的会员名单,包括他们的电子邮件、网站,并链接到他们的个人资料。'

# translate English to Chinese
tokenizer.src_lang = "zh_CN"
encoded_en = tokenizer(article_en, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_en,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

#Output: ['Yes, we can do that. We can extract the information you need from the Facebook group page and compile a list of members related to the holiday rental property, including their emails, websites and links to their personal information.']
```


