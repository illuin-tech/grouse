def get_positive_acceptance_negative_rejection(answer_relevancy, completeness):
    if answer_relevancy is None:
        if completeness is None:
            positive_acceptance = 1
            negative_rejection = 1
        else:
            positive_acceptance = 0
            negative_rejection = None
    else:
        if completeness is None:
            positive_acceptance = None
            negative_rejection = 0
        else:
            positive_acceptance = None
            negative_rejection = None
    return positive_acceptance, negative_rejection
