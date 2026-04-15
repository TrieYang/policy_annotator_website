from sklearn.metrics import cohen_kappa_score

rater1 = [2,2,3,2,5,5,5,5,5,3,5,5,4,3,5,5,5,5,5,5,5,5,4,5,5]  # my ratings
rater2 = [1,1,2,2,2,3,3,3,2,2,2,2,2,2,3,2,1,2,2,1,2,2,2,2,2]  # Claude

kappa_unweighted = cohen_kappa_score(rater1, rater2)

kappa_linear = cohen_kappa_score(rater1, rater2, weights="linear")

kappa_quadratic = cohen_kappa_score(rater1, rater2, weights="quadratic")

print("Unweighted Cohen’s Kappa:", kappa_unweighted)
print("Linear Weighted Cohen’s Kappa:", kappa_linear)
print("Quadratic Weighted Cohen’s Kappa:", kappa_quadratic)
