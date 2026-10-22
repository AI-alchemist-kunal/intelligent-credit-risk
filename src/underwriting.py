
def expected_credit_loss(pd, lgd, ead):
    return pd * lgd * ead


def expected_contribution(
    pd,
    lgd,
    ead,
    contribution_before_losses
):
    loss = expected_credit_loss(pd, lgd, ead)
    return contribution_before_losses - loss


if __name__ == "__main__":

    borrower_pd = 0.08
    lgd = 0.70
    ead = 100000
    margin = 8000

    loss = expected_credit_loss(
        borrower_pd, lgd, ead
    )

    contribution = expected_contribution(
        borrower_pd, lgd, ead, margin
    )

    print("Expected Credit Loss:", loss)
    print("Expected Contribution:", contribution)
