# Consider using gpt4_1k_2.json first.
The data gpt4_1k_1.json and gpt4_1k_2.json are generated from paragraphs of the first two 1k examples of InstructBlip_ver2.json.   
Some examples are filtered if the answer is not a Chinese poem.  
gpt4_1k_2.json is generated with prompts specifically asking for poem formatting, which results in some improvement of quality.
```  
The prompt for gpt4_1k_1.json is "請根據以下敘述生成一首詩詞：{paragraph}".  
The prompt for gpt4_1k_2.json is "根據以下的英文圖片敘述，請寫一首中文古詩。詩詞需要押韻，並保持每一句的字數一致。請讓詩詞的內容豐富，並帶有深遠的意境。圖片敘述：{paragraph}\n請根據這個敘述創作詩詞。"
```