import lorem


def generate_lorem_ipsum_paragraphs(num_paragraphs, sentences_per_paragraph):
    """Generate lorem ipsum text with the specified number
    of paragraphs and sentences per paragraph.

    Args:
        num_paragraphs (int): The number of paragraphs to generate.
        sentences_per_paragraph (int): The number of sentences in each paragraph.

    Returns:
        list: A list of generated lorem ipsum paragraphs.
    """
    lorem_paragraphs = []
    for _ in range(num_paragraphs):
        paragraph = [lorem.sentence() for _ in range(sentences_per_paragraph)]
        lorem_paragraphs.append(" ".join(paragraph))
    return lorem_paragraphs


def print_lorem_ipsum_paragraphs(lorem_paragraphs):
    """Print generated lorem ipsum paragraphs to the console.

    Args:
        lorem_paragraphs (list): A list of lorem ipsum paragraphs to print.
    """
    for index, paragraph in enumerate(lorem_paragraphs, start=1):
        print(f"Paragraph {index}:\n{paragraph}\n")


def main():
    num_paragraphs = 3
    sentences_per_paragraph = 4

    lorem_paragraphs = generate_lorem_ipsum_paragraphs(num_paragraphs, sentences_per_paragraph)
    print_lorem_ipsum_paragraphs(lorem_paragraphs)


if __name__ == "__main__":
    main()
