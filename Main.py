import os
from dotenv import load_dotenv 
from langchain_community.llms import HuggingFaceHub
from crewai import Agent, Task, Crew

load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceHub(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token = huggingface_api_key,
    task = "text-generation",
    model_kwargs = {"temperature":0.7, "max_length":512}
)

#Agents 
#Planner Agent
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article"
              "about the topic:{topic}."
              "You collect the information that helps the "
              "audience learn something"
              "and make informed decisions."
              "Your work is basis for"
              "the Content writer to write an article on this topic.",
    allow_delegation = False,
    verbose = True,
    llm = llm
)

#Writer Agent
writer = Agent(
    role="Content Writer",
    goal="Write a well structured article based on the content plan",
    backstory="You're working on a writing "
              "a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of "
              "the Content Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provide by the Content Planner. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provide by the Content Planner. "
              "You acknowledge in your opinion piece "
              "when your statements are opinions "
              "as opposed to objective statements.",
    allow_delegation= False,
    verbose = True,
    llm = llm
)

#Editor Agent
editor = Agent(
    role="Content Editor",
    goal="Improve Grammar, readability and overall quality of the article",
    backstory="You are an editor who receives a blog post "
              "from the Content Writer. "
              "Your goal is to review the blog post "
              "to ensure that it follows journalistic best practices,"
              "provides balanced viewpoints "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible"
              "and also look for grammar mistakes"
              "and also increase the readability of the article."
              "You are an experienced editor ensuring that the article is polished and professional.",
    allow_delegation=False,
    verbose= True,
    llm = llm
)

#Tasks
#Plan Task
plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
            "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
            "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
            "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document "
        "with an outline, audience analysis, "
        "SEO keywords, and resources.",
    agent=planner,
)

#Write Task
write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
            "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
		"3. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
            "engaging introduction, insightful body, "
            "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
    ),
    expected_output="A well-written blog post "
        "in markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs.",
    agent=writer, 
)

#Edit Task
edit = Task(
    description=(
        "Proofread the given blog post for "
        "grammatical errors and "
        "alignment with the brand's voice."
    ),
    expected_output = "A well-written blog post in markdown format, "
                    "ready for publication, "
                    "each section should have 2 or 3 paragraphs.",
    agent = editor
)

#creating the Crew
crew = Crew(
    agents=[planner,writer,editor],
    tasks=[plan,write,edit],
    verbose = True
)

#Get the topic from the user
topic = input("Enter a topic content generation:")
result = crew.kickoff(inputs={"topic":topic})

from IPython.display import Markdown
Markdown(result)