import assemblyai as aai
from dotenv import load_dotenv
import os

load_dotenv()

aai.settings.api_key = os.getenv("ASSEMBLY_API_KEY")

config = aai.TranscriptionConfig(
  speaker_labels=True,
  sentiment_analysis=True,
)
transcriber = aai.Transcriber()
audiofile = "./data/weaviate-podcast-109.mp3"
transcript = transcriber.transcribe(audiofile, config=config)
sentences = transcript.get_sentences()

num_sents = len(sentences)

print(num_sents)

with open('speaker_identification.txt', 'w') as file:
    for utterance in transcript.utterances:
        file.write(f"Speaker {utterance.speaker}: {utterance.text}\n")



with open('sentiment_analysis_output.txt', 'w') as file:
    for sentiment_result in transcript.sentiment_analysis:
        file.write(f"{sentiment_result.text}\n")
        file.write(f"{sentiment_result.sentiment}\n")  # POSITIVE, NEUTRAL, or NEGATIVE
        file.write(f"{sentiment_result.confidence}\n")
        file.write(f"Timestamp: {sentiment_result.start} - {sentiment_result.end}\n")


prompt1 = "How many times did they mention the word 'weaviate'?"
prompt2 = "Outline the main points of the podcast."
prompt3 = "Make a viral Linkedin post out of the podcast."
prompt4 = "Summarize the podcast as a christmas carol."

result1 = transcript.lemur.task(
   prompt1, final_model=aai.LemurModel.claude3_5_sonnet
)
print(result1.response)
result2 = transcript.lemur.task(
   prompt2, final_model=aai.LemurModel.claude3_5_sonnet
)
print(result2.response)

result3 = transcript.lemur.task(
    prompt3, final_model=aai.LemurModel.claude3_5_sonnet
    )
print(result3.response)
result4 = transcript.lemur.task(
    prompt4, final_model=aai.LemurModel.claude3_5_sonnet
    )
print(result4.response)
with open('full_transcript.txt', 'w') as file:
    file.write(f"{transcript.text}\n")
