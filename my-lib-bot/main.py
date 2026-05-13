from tools.embeddings import HuggingFaceEmbeddings


def get_documents_indexes(
    documents: list[str],
    embedding_model: str,
    parallel_processes: int = 1,
    run_inference_on: str = "cpu",
) -> list[tuple[int, str]]:
    """Get the index and document for each document in the input list."""

    with HuggingFaceEmbeddings(
        model=embedding_model,
        run_inference_on=run_inference_on,
        parallel_processes=parallel_processes,
    ) as embeddings_model:
        responses = embeddings_model.embed_documents(documents, show_progress_bar=True)

    return responses


if __name__ == "__main__":
    import pprint
    
    EMBEDDING_MODEL = "microsoft/harrier-oss-v1-0.6b"
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
        "A quiet library can feel like a small universe.",
        "The coffee tasted stronger than usual this morning.",
        "She packed her notebook before leaving the apartment.",
        "The old bridge shook slightly in the wind.",
        "A new bakery opened beside the train station.",
        "He forgot his umbrella during the afternoon storm.",
        "The meeting ended earlier than everyone expected.",
        "Fresh paint made the hallway smell unfamiliar.",
        "A child laughed while chasing bubbles in the park.",
        "The laptop fan hummed softly on the desk.",
        "They watched the sunset from the rooftop.",
        "The recipe called for too much garlic.",
        "A blue bicycle leaned against the garden fence.",
        "The museum was almost empty on Tuesday.",
        "She found an old postcard inside the book.",
        "The radio played a song from last summer.",
        "Clouds gathered above the distant mountains.",
        "He wrote the address on a yellow sticky note.",
        "The elevator stopped on the wrong floor.",
        "A friendly neighbor watered the plants.",
        "The market smelled like herbs and fresh bread.",
        "She solved the puzzle just before midnight.",
        "The dog slept under the kitchen table.",
        "A silver key was hidden beneath the mat.",
        "The train arrived five minutes late.",
        "He practiced guitar until his fingers hurt.",
        "The window reflected the city lights.",
        "A small boat crossed the quiet lake.",
        "She ordered soup because the evening was cold.",
        "The printer jammed during the final page.",
        "A red scarf hung from the chair.",
        "The garden path was covered with leaves.",
        "He bought a map before the road trip.",
        "The classroom smelled faintly of chalk.",
        "A candle flickered near the open window.",
        "She took notes during the science lecture.",
        "The phone rang twice and then stopped.",
        "A distant siren echoed through the street.",
        "The shopkeeper counted coins behind the counter.",
        "He discovered a typo in the report.",
        "The river moved slowly after the rain.",
        "A paper airplane landed near the doorway.",
        "She smiled after reading the message.",
        "The clock ticked loudly in the quiet room.",
        "A green apple rolled across the table.",
        "The hiking trail disappeared into the forest.",
        "He saved the file before closing the editor.",
    ]
    embedding_responses = get_documents_indexes(
        sentences,
        embedding_model=EMBEDDING_MODEL,
        parallel_processes=4,
        run_inference_on="cpu",
    )
    pprint.pprint(embedding_responses)