from personalized_model import (
    build_user_profile,
    personalized_recommendation
)

user_id = 1

liked = ["EX116", "EX132"]
saved = ["EX187"]
interests = ["Food & Culinary"]

user_vec = build_user_profile(
    user_key=user_id,
    liked_ids=liked,
    saved_ids=saved,
    interest_categories=interests
)

res = personalized_recommendation(user_id, user_vec)
print(res)
