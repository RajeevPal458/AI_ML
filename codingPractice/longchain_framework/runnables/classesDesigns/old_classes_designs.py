import random

class NakliLLM:

  def __init__(self):
    print('LLM created')

  def predict(self, prompt):

    response_list = [
        'Delhi is the capital of India',
        'IPL is a cricket league',
        'AI stands for Artificial Intelligence'
    ]

    return {'response': random.choice(response_list)}

class NakliPromptTemplate:

  def __init__(self, template, input_variables):
    self.template = template
    self.input_variables = input_variables

  def format(self, input_dict):
    return self.template.format(**input_dict)

template = NakliPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)

prompt = template.format({'length':'short','topic':'india'})
print(prompt)
llm = NakliLLM()

class NakliLLMChain:

  def __init__(self, llm, prompt):
    self.llm = llm
    self.prompt = prompt

  def run(self, input_dict):

    final_prompt = self.prompt.format(input_dict)
    result = self.llm.predict(final_prompt)

    return result['response']


template = NakliPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)

llm = NakliLLM()
chain = NakliLLMChain(llm, template)
result = chain.run({'length':'short', 'topic': 'india'})

print(f"result: {result}")


""""
So the conclusion is , old design classes was not generic and does not have common Interface
each feature classes has its own method to do the operation like PromptTemplate has formate() method while some has invoke() ....etc
that the reason We had many different types of Chains classes for different types of operations. 
chains was created to simply things and do the common operations in framework lets AI Engineers use then but at the end chains itself becomes overhead to Engineers to learn all those chains.

That why longchain team thought to redesigns the classes and have common method (invoke() ) to use by all concrete classes through common Runnable Interface 

let see new classes design how it solve this problems of chains (new file)
"""







