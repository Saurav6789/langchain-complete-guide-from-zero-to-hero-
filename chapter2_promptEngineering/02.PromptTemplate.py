from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOllama(model="qwen2.5-coder")
text = """
EU lawmakers on Thursday closed a deal to loosen laws under the EU’s Artificial Intelligence Act.
Details of the changes include postponing restrictions on high-risk uses of AI until December 2027 and exempting industrial applications of AI from legal scrutiny. AI tools designed to assist users will also not be considered under high-risk obligations, 
provided their malfunction doesn’t cause health and safety risks.
Companies will also be granted a three-month grace period on meeting new requirements to watermark AI-generated content.
Other stipulations include banning AI systems that create sexual abuse material, including those relating to children or non-consensual depictions of people engaged in sexual acts.

“With this agreement, we show that politics can move just as quickly as technology. We now make the AI rules more workable in practice, remove overlaps and pause the high-risk requirements,” Arba Kokalari of the EU’s Internal Market and Consumer Protection committee, said in a statement. 
"""
word_count = "150"
prompt_template = """

You are an expert in summarizing news. Please summarize the following text in {word_count} words.
Text: {text}
"""
summary_prompt = PromptTemplate.from_template(prompt_template)
summary = model.invoke(summary_prompt.format(word_count=word_count, text=text))
print("\n🟢 Template:", summary.content)
