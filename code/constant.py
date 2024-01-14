TASK_DESCRIPTION = {
    'question_generation': "Write some questions based on the given description.",
    'story_generation': "Write a story based on the given description.",
    'poem_generation': "Write a poem based on the given description.",
    "email_generation": "Write an email based on the given description.",
    'data_generation': "Generate data based on the given description.",
    "advice_giving": "Respond well to users when they seek advice.",
    "recommendations": "Give recommendations to users.",
    "how_to_generation": "Give relevant and complete answer when users ask `how to do` something.",
    "planning": "Write a plan for an event or activity.",
    "instructional_rewriting": "Rewrite a given text with a specific instruction.",
    "language_polishing": "Polish a piece of text to make it more fluent, natural, and readable.",
    "paraphrasing": "Paraphrase a given text.",
    "text_correction": "Correct the potential errors in a piece of text.",
    "code_correction": "Correct the potential errors in a piece of code.",
    "code_simplification": "Rewrite a piece of code to make it more concise and easy to understand.",
    "information_extraction": "Extract one or multiple user-specified categories of information from a piece of text attached in the user's query.",
    "keywords_extraction": "Extract the keywords from a piece of text.",
    'table_extraction': "Generate a table include the key information from a piece of text attached in the user's query.",
    "title_generation": "Generate a title for the given text or based on a description of the work.",
    "text_summarization": "Write a summary for a piece of text.",
    "note_summarization": "Write a note to summarize a piece of text.",
    "explain_code": "Write an explanation for a piece of code.",
    "explain_answer": "Explain something the user wants to know.",
    "text_to_text_translation": "Translate the given text into another language.",
    "text_to_code_translation": "Write a piece of code based on the given description.",
    "code_to_code_translation": "Convert the given code into another programming language.",
    "code_to_text_translation": "Write a document for the given code.",
    "open_qa": "The user's query is an open domain question with no attached passage or article. You should choose this if the query is a question and none of the other scenarios match it well.",
    "closed_qa": "Answer the questions that can be directly answered by the attached passage.",
    'fill_in_the_blank': "Complete the missing parts with the most appropriate words to make the text coherent and meaningful.",
    "fact_verification": "Verify if the given fact is true or false.",
    "math_puzzles": "Write an answer with the step-by-step reasoning process for a math question.",
    'language_learning_questions': "Write an answer for the given question about programming language learning.",
    'natural_language_learning_tutor': "Write an answer for the given question about natural language learning.",
    "exam_problem_solving_tutor": "Solve a science, technology or engineering exam question like fill-in-the-blank, multiple choice, true/false, matching, ordering, problem soving, etc.",
    'ml_ai_language_model_tutor': "Write an answer for the given question about machine learning, artificial intelligence or language model.",
    "general_classification": "Classify one or multiple objects given by the user into the specified categories.",
    "ordering": "Sort some things, according to some criteria.",
    'sentiment_analysis': "Identify and categorize the subjective opinions, attitudes, and feelings of the writer or speaker towards a particular subject, product, service, event, or entity.",
    'code_language_classification': "Classify the programming language for the given code.",
    'language_classification': "Classify the language for the given text.",
    "topic_classification": "Extract the high-level topics or themes from a given text, i.e., what kind of topics are discussed in the text.",
    "value_judgement": "Provide a value judgment on a given topic or statement.",
    "rejecting": "Reject to respond when the query is beyond capacity or it violates general ethical and legal rules.",
    "roleplay": "Pretend to be a specific person, character, profession or identity, and complete the required task on this basis.",
    "default": "You must choose this if none of the other scenarios match the user's query well.",
}


RAG_TASKS = [
    'how_to_generation',
    'open_qa',
    'fact_verification',
    'recommendations',
    'explain_answer'
]


REWRITE_TASK = [
    'question_generation',
    'email_generation',
    'data_generation',
    'recommendations',
    'how_to_generation',
    'planning',
    'instructional_rewriting',
    'language_polishing',
    'text_correction',
    'code_correction',
    'information_extraction',
    'keywords_extraction',
    'explain_code',
    'explain_answer',
    'text_to_text_translation',
    'text_to_code_translation',
    'code_to_code_translation',
    'code_to_text_translation',
    'open_qa',
    'closed_qa',
    'fill_in_the_blank',
    'fact_verification',
    'math_puzzles',
    'language_learning_questions',
    'natural_language_learning_tutor',
    'exam_problem_solving_tutor',
    'ml_ai_language_model_tutor',
    'general_classification',
    'ordering',
    'sentiment_analysis',
    'code_language_classification',
    'language_classification',
    'topic_classification',
    'value_judgement',
    'rejecting',
]


REWRITING_SYSTEM_PROMPT: str = '''
Please act as a rewriter to modify the format of the AI assistant's response to the user's question presented below.

Please follow the instructions below:
1. Please first determine whether the given format meets the requirements of the user's question, if it does not, then copy the AI assistant's response, if it does, then modify the response's format following the provided format.
2. Your task is limited to altering the format while keeping the original meaning and information intact.
3. Please make sure that the revised response can answer the user's question correctly.
4. Please make sure that the revised response is fluent and has no additional subheadings.

Please first write "Reasoning: <reason>" to provide a brief reasoning you used to modify, and then write "Revised response: <response>" to output your final revised response without any additional information, ensuring its fluency.
Do not output any additional subheadings.
'''

REWRITING_USER_PROMPT: str = '''
Below is a user's question, the AI assistant's response, and the provided format. 

[Question start]
{question}
[Question end]

[Response start]
{response}
[Response end]

[Format start]
{structure}
[Format end]
'''

REWRITING_RETRIEVAL_SYSTEM_PROMPT: str = '''
Please act as a rewriter to modify the format of the AI assistant's response to the user's question presented below.

Please follow the instructions below:
1. Please first determine whether the given format meets the requirements of the user's question, if it does not, then copy the AI assistant's response, if it does, then modify the response's format following the provided format.
2. Your task is limited to altering the format while keeping the original meaning and information intact.
3. Please make sure that the revised response can answer the user's question correctly.
4. Please make sure that the revised response is fluent and has no additional subheadings.
5. Evidence is the useful information. You should decide for yourself which parts of the evidence to help rewriting the response.

Please first write "Reasoning: <reason>" to provide a brief reasoning you used to modify, and then write "Revised response: <response>" to output your final revised response without any additional information, ensuring its fluency.
Do not output any additional subheadings.
'''

REWRITING_RETRIEVAL_USER_PROMPT: str = '''
Below is a user's question, the AI assistant's response, the provided format, and the evidences. 

[Question start]
{question}
[Question end]

[Response start]
{response}
[Response end]

[Format start]
{structure}
[Format end]

[Evidence start]
{evidence}
[Evidence end]
'''

STRUCTURE: dict = {
    'question_generation': '''It is a question-generating task. Use a list to give the generated questions.''',

    "email_generation": '''It is an email-writing task. Here is a general guideline for creating a well-structured and professional email:
1. Subject Line: Write a clear and concise subject line that accurately summarizes the content of your email. This helps the recipient understand the purpose of the email at a glance.
2. Salutation: Begin your email with a formal salutation such as "Dear [Recipient's Name]," or use a more casual salutation if you have an informal relationship with the recipient.
3. Introduction: Start your email with a brief introduction, stating who you are and the reason for writing the email. Be clear and to the point, and avoid unnecessary details.
4. Body: This is the main content of your email. Organize your thoughts into paragraphs or bullet points to make them easier to read. Keep your sentences concise and focused. Use proper grammar, punctuation, and spelling to maintain professionalism. If you need to discuss multiple topics, consider using headings or numbered points to separate them.
5. Politeness and Tone: Maintain a polite and respectful tone throughout your email. Be mindful of the recipient's perspective and use appropriate language. Avoid using excessive capitalization, exclamation marks, or emoticons, as they can come across as unprofessional.
6. Closing: Conclude your email with a closing remark, such as "Thank you," or "Best regards," followed by your name. If you expect a response or need specific action, you can mention it in this section as well.
7. Signature: Include your full name, job title, and contact information (e.g., phone number, email address) in your email signature. This helps the recipient easily identify and contact you if needed.
8. Attachments: If you need to include attachments, mention them in the email body and make sure they are relevant to the email's purpose. Consider compressing large files or using cloud storage services if the attachments are too large to be sent via email.
9. Proofread: Before sending the email, proofread it for any grammatical or spelling errors. Make sure the email conveys your message clearly and effectively.
The best emails are short, direct, professional, and scannable for the recipient. Follow formal business email structure unless you have an established casual rapport with the recipient.''',

    "data_generation": '''It is a data-generating task. Use a list to give the generated data.''',

    "recommendations": '''This is a task for giving recommendations. The first sentence should identify the intended purpose of the individual you are recommending. Afterward, give the recommendations to meet these objectives. Then, use a list to give the explanations. Last, give a conclusion.''',

    "how_to_generation": '''This is a how-to question. First, analyze the question. Then, give the answer. Next, give the corresponding explanations. Last, give a conclusion.''',

    "planning": '''This is a plan-writing task. First, give the planning goals in the initial sentence. Afterward, Use a list to outline the plan by timeline. Then, give the explanations. Last, give a conclusion.''',

    "instructional_rewriting": '''This is a guided rewrite task. First, output the rewritten content. And then output a list to present what has been changed and why it changed''',

    "language_polishing": '''It's a language polishing task. First, output the polished content. And then output a list to present what has been changed and why it changed.''',

    "text_correction": '''This is a text correction task. First, output the corrected content. And then output a list to present what has been changed and why it changed.''',

    "code_correction": '''This is a code correction task. First, output the corrected code and its corresponding comments within a code block. And then output a list to present what has been changed and why it changed.''',

    "information_extraction": '''This is an information extraction task. Use a structured format to give the extracted information, such as a list or table.''',

    "keywords_extraction": '''This is a keywords extraction task. Use a list to give the keywords.''',

    "explain_code": '''This is a code explanation task. First, analyze the question and give a brief analysis in the first paragraph. Then, a structured format to give the explanation such as a list or table. Last, give a conclusion.''',

    "explain_answer": '''This is an explanation task. First, analyze the question and give a brief analysis in the first paragraph. Then, a structured format to give the explanation such as a list or table. Last, give a conclusion.''',

    "text_to_text_translation": '''This is a translation task, please give the translated content first and then use a list to give an explanation.''',

    "text_to_code_translation": '''This is a task to write code based on text requirements. First, analyze the question and give a brief analysis in the first paragraph. Next, output the code and corresponding comments in a code block. Then, use a list to give the explanation for each piece of code. Last, give a conclusion.''',

    "code_to_code_translation": '''This is a task to convert the given code into another programming language. First, analyze the question and give a brief analysis in the first paragraph. Next, output the converted code and corresponding comments in a code block. Then, use a list to give the explanation for each piece of code. Last, give a conclusion.''',

    "code_to_text_translation": '''This is a task to write a document for the given code. First, analyze the question and give a brief analysis in the first paragraph. Next, take a segmented and structured way to write this document. Last, give a conclusion.''',

    "open_qa": '''This is an open-ended question-and-answer task. First, analyze the question and give a brief analysis in the first paragraph. Next, give the answer. Then, give an explanation. Last, give a conclusion.''',

    "closed_qa": '''This is a task to answer the questions that can be directly answered by the attached passage. First, analyze the question and give a brief analysis in the first paragraph. Next, give the answer. Then, give an explanation. Last, give a conclusion.''',

    'fill_in_the_blank': '''This is a task to complete the missing parts with the most appropriate words to make the text coherent and meaningful. First, analyze the question and give a brief analysis in the first paragraph. Then, give the answer and an explanation.''',

    "fact_verification": '''This is a fact-verification task. First, give the answer. Then, give an explanation.''',

    "math_puzzles": '''This is a math question. First, analyze the question and give a brief analysis in the first paragraph. Then, use a list to present the step-by-step solution. Next, give another list to output a detailed explanation. Last, give the correct result and a conclusion.''',

    'language_learning_questions': '''This is a task to answer the given question about programming language learning. First, analyze the question and give a brief analysis in the first paragraph. Then output the answer. Next, give an explanation. Last, give a conclusion.''',

    'natural_language_learning_tutor': '''This is a task to answer the given question about natural language learning. First, analyze the question and give a brief analysis in the first paragraph. Then output the answer. Next, give an explanation. Last, give a conclusion.''',

    "exam_problem_solving_tutor": '''This is an exam problem. First, analyze the question and give a brief analysis in the first paragraph. Then output the answer. Next, give an explanation. Last, give a conclusion.''',

    'ml_ai_language_model_tutor': '''This is a question about machine learning, artificial intelligence or language model. Then output the answer. Next, give an explanation. Last, give a conclusion.''',

    "general_classification": '''This is a classification task. First, answer. Then, explain.''',

    "ordering": '''This is an ordering question. First, answer. Then, explain.''',

    'sentiment_analysis': '''This is a sentiment analysis question. First, answer. Then, explain.''',

    'code_language_classification': '''This is a task to classify the programming language for the given code. First, answer. Then, explain.''',

    'language_classification': '''This is a task to classify the language for the given text. First, answer. Then, explain.''',

    "topic_classification": '''This is a task to extract the high-level topics or themes from a given text. First, answer. Then, explain.''',

    "value_judgement": '''This is a value judgment task. First, analyze the question and give a brief analysis in the first paragraph. Then output the answer. Next, give an explanation. Last, give a conclusion.''',

    "rejecting": '''This question should be rejected to answer. First, reject. Then, explain why to reject.''',

    "default": '''First, analyse the question and give a brief analysis in the first paragraph. Then output the answer. Next, use a list to give explanations. Last, give a conclusion.'''
}


task_group = {
    "Generation": [
        'question_generation', 'story_generation', 'poem_generation', "email_generation", 'data_generation', "text_to_text_translation"
    ],
    'Brainstorming': [
        "advice_giving", "recommendations", "how_to_generation", "planning",
    ],
    'Code': [
        "code_correction", "code_simplification", "explain_code", "text_to_code_translation", "code_to_code_translation", 'language_learning_questions', 'code_language_classification', "code_to_text_translation"
    ],
    'Rewriting': [
        "instructional_rewriting", "language_polishing", "paraphrasing", "text_correction",
    ],
    'Extraction': [
        "information_extraction", "keywords_extraction", 'table_extraction',
    ],
    'Summarization': [
        "title_generation", "text_summarization", "note_summarization"
    ],
    'Conversation': [
        "open_qa", "closed_qa", "fact_verification", "value_judgement", "roleplay", "explain_answer"
    ],
    'Specialized Educational Dialog': [
        'natural_language_learning_tutor', "exam_problem_solving_tutor", 'ml_ai_language_model_tutor', "math_puzzles", 'fill_in_the_blank'
    ],
    'Classification': [
        "general_classification", "ordering", 'sentiment_analysis', 'language_classification', "topic_classification"
    ],
    'Others': [
        'default', 'rejecting'
    ]
}

READABILITY_EVALUATION_SYSTEM_PROMPT: str = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider the readability including clarity, format, and ease of understanding. Well-organized, grammatically correct response is better. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."
READABILITY_EVALUATION_USER_PROMPT: str = "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"

FACTUALITY_EVALUATION_SYSTEM_PROMPT: str = "Please act as an impartial judge and evaluate the factuality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\"."
FACTUALITY_EVALUATION_USER_PROMPT: str = "[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"

NON_ADAPTIVITTY_REWRITING_SYSTEM_PROMPT: str = '''
Please act as a rewriter to modify the format of the AI assistant's response to the user's question presented below.

Please follow the instructions below:
1. Your task is limited to altering the format while keeping the original meaning and information intact.
2. Please make sure that the revised response can answer the user's question correctly.
3. Please make sure that the revised response is fluent and has no additional subheadings.

Please first write "Reasoning: <reason>" to provide a brief reasoning you used to modify, and then write "Revised response: <response>" to output your final revised response without any additional information, ensuring its fluency.
Do not output any additional subheadings.
'''

NON_ADAPTIVITTY_REWRITING_RETRIEVAL_SYSTEM_PROMPT: str = '''
Please act as a rewriter to modify the format of the AI assistant's response to the user's question presented below.

Please follow the instructions below:
1. Your task is limited to altering the format while keeping the original meaning and information intact.
2. Please make sure that the revised response can answer the user's question correctly.
3. Please make sure that the revised response is fluent and has no additional subheadings.
4. Evidence is the useful information. You should decide for yourself which parts of the evidence to help rewriting the response.

Please first write "Reasoning: <reason>" to provide a brief reasoning you used to modify, and then write "Revised response: <response>" to output your final revised response without any additional information, ensuring its fluency.
Do not output any additional subheadings.
'''
