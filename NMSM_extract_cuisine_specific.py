# -*- coding: utf-8 -*-

def get_bussiness_ids(category):
    '''Gets all the (anonynimized)business ids for a given category'''
    with open(business_fn) as businesses:
        business_ids = []
        for business in businesses:
            business = json.loads(business)
            if category in business['categories']:
                business_ids.append(business['business_id'])
    return business_ids
    
business_fn = 'yelp_business_ids.json'
reviews_fn = 'yelp_restaurant_review.json'
template_fn = 'mexican_restaurant_reviews.json' # replace with thai_restaurant_reviews for creating the Thai file

def save_reviews(category, cuisine):
    '''Saves the given number of reviews of a specific category to two files, 
    one for each class(pos/neg).'''
    business_ids = get_bussiness_ids(category)
    
    res_reviews = open(template_fn, 'w')
    
    with open(reviews_fn) as reviews:
        for review in reviews:
            # stop when quantity is reached
            review = json.loads(review)
            if review['business_id'] in business_ids and cuisine in review['text']:
                json.dump(review, res_reviews)
                res_reviews.write('\n')

save_reviews('Restaurants', "Mexican")

           
   
   
