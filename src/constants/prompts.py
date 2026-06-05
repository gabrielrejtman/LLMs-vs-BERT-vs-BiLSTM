"""Prompt constants for sentiment classification experiments."""

few_shot_balanced = """
Review: "a visually stunning and moving film." Sentiment: positive
Review: "an absolute joy to watch." Sentiment: positive
Review: "masterful storytelling." Sentiment: positive
Review: "a dull, lifeless waste of time." Sentiment: negative
Review: "fails on almost every level." Sentiment: negative"""

few_shot_contrastive = """
Review: "not a bad movie by any stretch." Sentiment: positive
Review: "I've had more fun watching paint dry." Sentiment: negative
Review: "It lacks the charm of the original." Sentiment: negative
Review: "surprisingly good despite the terrible trailer." Sentiment: positive
Review: "a masterclass in how not to make a thriller." Sentiment: negative"""

few_shot_diverse = """
Review: "Great!" Sentiment: positive
Review: "The acting was wooden and the plot made absolutely no sense." Sentiment: negative
Review: "A masterpiece." Sentiment: positive
Review: "While the cinematography is decent, the characters are entirely forgettable and the pacing drags terribly." Sentiment: negative
Review: "Loved it." Sentiment: positive"""

prompt_templates = {
    "ZS-1": 'Classify the sentiment of the following movie review as positive or negative.\nReview: "{review_text}"\nSentiment:',
    "ZS-2": 'A positive sentiment expresses satisfaction, enjoyment, or approval. A negative sentiment expresses dissatisfaction, disappointment, or disapproval. Based on these definitions, classify the sentiment of the following movie review as positive or negative.\nReview: "{review_text}"\nSentiment:',
    "ZS-3": 'Consider the overall emotional tone of the following movie review. Does the reviewer feel positively or negatively about the movie?\nReview: "{review_text}"\nEmotional tone (positive/negative):',
    "FS-1": f'Classify the sentiment of the following movie review as positive or negative.\nExamples:{few_shot_balanced}\nReview: "{{review_text}}"\nSentiment:',
    "FS-2": f'Classify the sentiment of the following movie review as positive or negative. Pay attention to negation and sarcasm.\nExamples:{few_shot_contrastive}\nReview: "{{review_text}}"\nSentiment:',
    "FS-3": f'Classify the sentiment of the following movie review as positive or negative.\nExamples:{few_shot_diverse}\nReview: "{{review_text}}"\nSentiment:',
    "CoT-1": 'Analyze the sentiment of the following movie review step by step. Consider the overall tone, key phrases, and any negation or contrast. Then classify as positive or negative. End your response with "Final Classification: positive" or "Final Classification: negative".\nReview: "{review_text}"\nStep-by-step analysis:',
    "CoT-2": 'Analyze the sentiment of the following movie review by answering: (1) What is the overall tone? (2) Are there key positive or negative phrases? (3) Is there negation or contrast? Then classify as positive or negative. End your response with "Final Classification: positive" or "Final Classification: negative".\nReview: "{review_text}"\nAnalysis:',
    "CoT-3": 'Before classifying the sentiment of the following movie review, ask yourself: Would most readers interpret this as positive or negative? Why? Then provide your classification. End your response with "Final Classification: positive" or "Final Classification: negative".\nReview: "{review_text}"\nReasoning:'
}
