# app.py - Enhanced Recommendation Server with Tag-based Filtering
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import os
from datetime import datetime, timedelta
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging
import threading
import time
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from collections import Counter


# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Environment variables loaded from .env file")
except ImportError:
    print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")
    print("⚠ Falling back to system environment variables")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# MongoDB connection with better error handling
def initialize_mongodb():
    """Initialize MongoDB connection with proper error handling"""
    try:
        MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/nft_marketplace')
        
        # Log connection attempt (hide sensitive info)
        safe_uri = MONGODB_URI
        if '@' in MONGODB_URI:
            # Hide credentials in logs
            parts = MONGODB_URI.split('@')
            safe_uri = parts[0].split('://')[0] + '://***:***@' + parts[1]
        
        logger.info(f"Attempting to connect to MongoDB: {safe_uri}")
        
        # Create MongoDB client with appropriate settings
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=15000,  # 15 seconds
            connectTimeoutMS=10000,          # 10 seconds
            socketTimeoutMS=10000,           # 10 seconds
            retryWrites=True,
            w='majority'
        )
        
        # Test the connection
        client.admin.command('ping')
        logger.info("✓ Successfully connected to MongoDB")
        
        # Get database - explicitly specify the database name
        # Since your MongoDB database is named "test", we'll use that
        db = client['test']  # or client.test
        logger.info(f"✓ Connected to database: {db.name}")
        
        # Test database access by listing collections
        try:
            collections = db.list_collection_names()
            logger.info(f"✓ Available collections: {collections}")
        except Exception as e:
            logger.warning(f"Could not list collections: {e}")
        
        return client, db
        
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        raise

# Initialize MongoDB connection
try:
    client, db = initialize_mongodb()
    
    # Initialize collections
    interactions_collection = db.interactions
    nfts_collection = db.nfts
    recommendations_collection = db.recommendations
    users_collection = db.users
    batch_jobs_collection = db.batchjobs
    
    logger.info("✓ All database collections initialized")
    
except Exception as e:
    logger.error(f"❌ Failed to initialize database: {e}")
    logger.error("Application cannot start without database connection")
    exit(1)

class NFTRecommendationEngine:
    def __init__(self):
        self.interaction_weights = {
            'view': 1.0,
            'like': 3.0,
            'bid': 5.0,
            'purchase': 10.0,
            'create': 2.0
        }
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        
    def get_nft_by_id(self, nft_id):
        """Get NFT data by ID"""
        try:
            nft = nfts_collection.find_one({'tokenId': str(nft_id)})
            return nft
        except Exception as e:
            logger.error(f"Error fetching NFT {nft_id}: {e}")
            return None
    
    def get_user_interactions(self, user_id):
        """Get user's interaction history"""
        try:
            interactions = list(interactions_collection.find({
                'userId': ObjectId(user_id)
            }).sort('timestamp', -1).limit(1000))
            return interactions
        except Exception as e:
            logger.error(f"Error fetching user interactions: {e}")
            return []
    
    def get_user_profile(self, user_id):
        """Build user preference profile from interactions using NFT tags"""
        interactions = self.get_user_interactions(user_id)
        
        if not interactions:
            return {
                'preferredCategories': [],  # Will be tag-based categories
                'preferredPriceRange': 'medium',
                'avgInteractionScore': 0,
                'preferredTags': []
            }
        
        # Analyze tags from NFTs
        tag_scores = {}
        price_interactions = []
        total_score = 0
        
        for interaction in interactions:
            weight = self.interaction_weights.get(interaction['interactionType'], 1.0)
            total_score += weight
            
            # Get tags from NFT or interaction metadata
            tags = []
            
            # First, try to get tags from interaction metadata
            if 'metadata' in interaction and 'tags' in interaction['metadata']:
                tags = interaction['metadata']['tags']
            else:
                # If not in interaction, fetch from NFT document
                nft = self.get_nft_by_id(interaction.get('nftId'))
                if nft and 'tags' in nft:
                    tags = nft['tags']
            
            # Score the tags
            for tag in tags:
                if tag:  # Make sure tag is not empty
                    tag_scores[tag] = tag_scores.get(tag, 0) + weight
            
            # Price range analysis
            price = None
            if 'metadata' in interaction and 'price' in interaction['metadata']:
                try:
                    price = float(interaction['metadata']['price'])
                except (ValueError, TypeError):
                    pass
            
            # If price not in interaction metadata, get from NFT
            if price is None:
                nft = self.get_nft_by_id(interaction.get('nftId'))
                if nft and 'price' in nft:
                    try:
                        price = float(nft['price'])
                    except (ValueError, TypeError):
                        pass
            
            if price is not None:
                price_interactions.append(price)
        
        # Get preferred tags (top 5)
        preferred_tags = sorted(tag_scores.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        preferred_tags = [tag[0] for tag in preferred_tags]
        
        # Create categories based on tags (group similar tags)
        preferred_categories = self.group_tags_into_categories(preferred_tags)
        
        # Determine price range preference
        preferred_price_range = 'medium'
        if price_interactions:
            avg_price = np.mean(price_interactions)
            if avg_price < 0.1:
                preferred_price_range = 'low'
            elif avg_price < 1.0:
                preferred_price_range = 'medium'
            elif avg_price < 10.0:
                preferred_price_range = 'high'
            else:
                preferred_price_range = 'premium'
        
        return {
            'preferredCategories': preferred_categories,
            'preferredPriceRange': preferred_price_range,
            'avgInteractionScore': total_score / len(interactions) if interactions else 0,
            'preferredTags': preferred_tags
        }
    
    def group_tags_into_categories(self, tags):
        """Group tags into meaningful categories"""
        # Define tag categories
        category_mappings = {
            'Art': ['art', 'painting', 'drawing', 'artwork', 'artistic', 'creative', 'design'],
            'Gaming': ['game', 'gaming', 'character', 'weapon', 'avatar', 'rpg', 'fantasy'],
            'Music': ['music', 'audio', 'sound', 'song', 'album', 'artist', 'musician'],
            'Sports': ['sport', 'football', 'basketball', 'soccer', 'athlete', 'team'],
            'Collectibles': ['collectible', 'rare', 'limited', 'exclusive', 'vintage', 'unique'],
            'Technology': ['tech', 'ai', 'robot', 'cyber', 'digital', 'blockchain', 'crypto'],
            'Nature': ['nature', 'animal', 'landscape', 'tree', 'flower', 'ocean', 'mountain'],
            'Fashion': ['fashion', 'style', 'clothing', 'accessory', 'luxury', 'brand'],
            'Photography': ['photo', 'photography', 'portrait', 'street', 'landscape', 'macro'],
            'Abstract': ['abstract', 'geometric', 'pattern', 'minimalist', 'modern']
        }
        
        categories = []
        for tag in tags:
            tag_lower = tag.lower()
            for category, keywords in category_mappings.items():
                if any(keyword in tag_lower for keyword in keywords):
                    if category not in categories:
                        categories.append(category)
                    break
        
        return categories[:3]  # Return top 3 categories
    
    def get_similar_nfts(self, target_nft, all_nfts, top_k=10):
        """Find similar NFTs based on tags and content features"""
        try:
            # Prepare text features including tags
            texts = []
            nft_ids = []
            
            for nft in all_nfts:
                text_features = []
                text_features.append(nft.get('name', ''))
                text_features.append(nft.get('description', ''))
                
                # Add tags to text features
                if 'tags' in nft and nft['tags']:
                    text_features.extend(nft['tags'])
                
                texts.append(' '.join(text_features))
                nft_ids.append(nft['tokenId'])
            
            if len(texts) < 2:
                return []
            
            # TF-IDF vectorization
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Find target NFT index
            target_index = None
            for i, nft_id in enumerate(nft_ids):
                if nft_id == target_nft['tokenId']:
                    target_index = i
                    break
            
            if target_index is None:
                return []
            
            # Calculate similarities
            similarities = cosine_similarity(tfidf_matrix[target_index:target_index+1], 
                                           tfidf_matrix).flatten()
            
            # Get top similar NFTs (excluding the target itself)
            similar_indices = similarities.argsort()[::-1][1:top_k+1]
            
            similar_nfts = []
            for idx in similar_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    similar_nfts.append({
                        'nftId': nft_ids[idx],
                        'score': float(similarities[idx]),
                        'reason': f"Similar to {target_nft['name']}"
                    })
            
            return similar_nfts
            
        except Exception as e:
            logger.error(f"Error calculating similar NFTs: {e}")
            return []
    
    def get_collaborative_recommendations(self, user_id, user_profile, all_nfts):
        """Get recommendations based on collaborative filtering"""
        try:
            # Find users with similar interaction patterns
            user_interactions = self.get_user_interactions(user_id)
            user_nft_ids = set(i.get('nftId') for i in user_interactions if i.get('nftId'))
            
            if not user_nft_ids:
                return []
            
            # Find other users who interacted with similar NFTs
            similar_users = []
            other_user_interactions = list(interactions_collection.aggregate([
                {'$match': {'userId': {'$ne': ObjectId(user_id)}}},
                {'$group': {
                    '_id': '$userId',
                    'nftIds': {'$addToSet': '$nftId'},
                    'interactions': {'$push': '$$ROOT'}
                }}
            ]))
            
            for other_user in other_user_interactions:
                other_nft_ids = set(other_user['nftIds'])
                overlap = len(user_nft_ids.intersection(other_nft_ids))
                
                if overlap > 0:
                    similarity = overlap / len(user_nft_ids.union(other_nft_ids))
                    if similarity > 0.1:  # Minimum similarity threshold
                        similar_users.append({
                            'userId': other_user['_id'],
                            'similarity': similarity,
                            'interactions': other_user['interactions']
                        })
            
            # Get recommendations from similar users
            recommendations = {}
            for similar_user in similar_users[:10]:  # Top 10 similar users
                for interaction in similar_user['interactions']:
                    nft_id = interaction.get('nftId')
                    if nft_id and nft_id not in user_nft_ids:
                        weight = (self.interaction_weights.get(interaction['interactionType'], 1.0) * 
                                similar_user['similarity'])
                        recommendations[nft_id] = recommendations.get(nft_id, 0) + weight
            
            # Sort and return top recommendations
            collab_recs = []
            for nft_id, score in sorted(recommendations.items(), 
                                      key=lambda x: x[1], reverse=True)[:20]:
                collab_recs.append({
                    'nftId': nft_id,
                    'score': float(score),
                    'reason': "Users with similar taste also liked this"
                })
            
            return collab_recs
            
        except Exception as e:
            logger.error(f"Error in collaborative filtering: {e}")
            return []
    
    def get_trending_recommendations(self, user_profile, all_nfts):
        """Get trending NFTs based on recent interactions"""
        try:
            # Get recent interactions (last 7 days)
            week_ago = datetime.now() - timedelta(days=7)
            
            trending_pipeline = [
                {'$match': {'timestamp': {'$gte': week_ago}}},
                {'$group': {
                    '_id': '$nftId',
                    'total_interactions': {'$sum': 1},
                    'likes': {'$sum': {'$cond': [{'$eq': ['$interactionType', 'like']}, 1, 0]}},
                    'views': {'$sum': {'$cond': [{'$eq': ['$interactionType', 'view']}, 1, 0]}},
                    'purchases': {'$sum': {'$cond': [{'$eq': ['$interactionType', 'purchase']}, 1, 0]}}
                }},
                {'$addFields': {
                    'trending_score': {
                        '$add': [
                            {'$multiply': ['$views', 1]},
                            {'$multiply': ['$likes', 3]},
                            {'$multiply': ['$purchases', 10]}
                        ]
                    }
                }},
                {'$sort': {'trending_score': -1}},
                {'$limit': 50}
            ]
            
            trending_nfts = list(interactions_collection.aggregate(trending_pipeline))
            
            recommendations = []
            for trending in trending_nfts:
                # Filter by user preferences
                nft = nfts_collection.find_one({'tokenId': trending['_id']})
                if nft and nft.get('isActive', True):
                    # Check tag preference
                    nft_tags = nft.get('tags', [])
                    user_tags = user_profile.get('preferredTags', [])
                    
                    # If user has tag preferences, check if NFT has matching tags
                    if not user_tags or any(tag in nft_tags for tag in user_tags):
                        recommendations.append({
                            'nftId': trending['_id'],
                            'score': float(trending['trending_score']) / 100.0,  # Normalize
                            'reason': "Trending this week"
                        })
            
            return recommendations[:10]
            
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    def get_tag_based_recommendations(self, user_profile, all_nfts):
        """Get recommendations based on user's preferred tags"""
        try:
            preferred_tags = user_profile.get('preferredTags', [])
            if not preferred_tags:
                return []
            
            recommendations = []
            
            # Find NFTs with matching tags
            for nft in all_nfts:
                if not nft.get('isActive', True):
                    continue
                    
                nft_tags = nft.get('tags', [])
                if not nft_tags:
                    continue
                
                # Calculate tag match score
                matching_tags = set(preferred_tags).intersection(set(nft_tags))
                if matching_tags:
                    # Score based on number of matching tags and NFT popularity
                    tag_score = len(matching_tags) / len(preferred_tags)
                    popularity_score = (nft.get('totalLikes', 0) + nft.get('totalViews', 0)) / 100.0
                    
                    final_score = (tag_score * 0.7) + (popularity_score * 0.3)
                    
                    recommendations.append({
                        'nftId': nft['tokenId'],
                        'score': float(final_score),
                        'reason': f"Matches your interests: {', '.join(list(matching_tags)[:2])}"
                    })
            
            # Sort by score and return top recommendations
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:15]
            
        except Exception as e:
            logger.error(f"Error getting tag-based recommendations: {e}")
            return []
    
    def generate_recommendations_for_user(self, user_id):
        """Generate comprehensive recommendations for a user"""
        try:
            # Get user profile
            user_profile = self.get_user_profile(user_id)
            
            # Get all active NFTs
            all_nfts = list(nfts_collection.find({'isActive': True}).limit(1000))
            
            if not all_nfts:
                return []
            
            all_recommendations = []
            
            # 1. Content-based recommendations (based on user's liked/purchased NFTs)
            user_interactions = self.get_user_interactions(user_id)
            liked_nfts = [i for i in user_interactions if i['interactionType'] in ['like', 'purchase']]
            
            for interaction in liked_nfts[:5]:  # Top 5 liked/purchased NFTs
                nft = nfts_collection.find_one({'tokenId': interaction.get('nftId')})
                if nft:
                    similar_nfts = self.get_similar_nfts(nft, all_nfts, top_k=5)
                    for sim_nft in similar_nfts:
                        sim_nft['confidence'] = 0.8
                    all_recommendations.extend(similar_nfts)
            
            # 2. Tag-based recommendations
            tag_recs = self.get_tag_based_recommendations(user_profile, all_nfts)
            for rec in tag_recs:
                rec['confidence'] = 0.7
            all_recommendations.extend(tag_recs)
            
            # 3. Collaborative filtering
            collab_recs = self.get_collaborative_recommendations(user_id, user_profile, all_nfts)
            for rec in collab_recs:
                rec['confidence'] = 0.6
            all_recommendations.extend(collab_recs)
            
            # 4. Trending recommendations
            trending_recs = self.get_trending_recommendations(user_profile, all_nfts)
            for rec in trending_recs:
                rec['confidence'] = 0.5
            all_recommendations.extend(trending_recs)
            
            # Remove duplicates and sort
            seen_nfts = set()
            unique_recommendations = []
            user_interacted_nfts = set(i.get('nftId') for i in user_interactions)
            
            for rec in all_recommendations:
                nft_id = rec['nftId']
                if nft_id not in seen_nfts and nft_id not in user_interacted_nfts:
                    seen_nfts.add(nft_id)
                    unique_recommendations.append(rec)
            
            # Sort by score and return top 50
            unique_recommendations.sort(key=lambda x: x['score'], reverse=True)
            return unique_recommendations[:50]
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return []

# Initialize recommendation engine
rec_engine = NFTRecommendationEngine()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/recommendations/by-wallet/<wallet_address>', methods=['GET'])
def get_recommendations_by_wallet(wallet_address):
    """Get recommendations for a user by wallet address"""
    try:
        limit = int(request.args.get('limit', 10))
        
        # Find user by wallet address
        user = users_collection.find_one({'walletAddress': wallet_address.lower()})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        user_id = str(user['_id'])
        
        # Check if recommendations exist and are not expired
        recommendation = recommendations_collection.find_one({'userId': ObjectId(user_id)})
        
        if (recommendation and 
            recommendation.get('expiresAt', datetime.min) > datetime.now()):
            
            # Return cached recommendations
            top_recommendations = recommendation['recommendations'][:limit]
            return jsonify({
                'recommendations': top_recommendations,
                'userProfile': recommendation.get('userProfile', {}),
                'lastCalculated': recommendation.get('lastCalculated', datetime.now()),
                'source': 'cached'
            })
        
        # Generate new recommendations if not cached or expired
        recommendations = rec_engine.generate_recommendations_for_user(user_id)
        user_profile = rec_engine.get_user_profile(user_id)
        
        # Cache the recommendations
        recommendations_collection.update_one(
            {'userId': ObjectId(user_id)},
            {
                '$set': {
                    'userId': ObjectId(user_id),
                    'walletAddress': user['walletAddress'],
                    'recommendations': recommendations,
                    'userProfile': user_profile,
                    'lastCalculated': datetime.now(),
                    'expiresAt': datetime.now() + timedelta(hours=6)
                }
            },
            upsert=True
        )
        
        top_recommendations = recommendations[:limit]
        
        return jsonify({
            'recommendations': top_recommendations,
            'userProfile': user_profile,
            'lastCalculated': datetime.now(),
            'source': 'fresh'
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations by wallet: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/recommendations/refresh/<wallet_address>', methods=['POST'])
def refresh_recommendations_by_wallet(wallet_address):
    """Force refresh recommendations for a specific user"""
    try:
        # Find user by wallet address
        user = users_collection.find_one({'walletAddress': wallet_address.lower()})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        user_id = str(user['_id'])
        
        # Delete cached recommendations to force refresh
        recommendations_collection.delete_one({'userId': ObjectId(user_id)})
        
        # Generate new recommendations
        return get_recommendations_by_wallet(wallet_address)
        
    except Exception as e:
        logger.error(f"Error refreshing recommendations: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch/generate-all', methods=['POST'])
def batch_generate_all_recommendations():
    """Generate recommendations for all users in batches"""
    try:
        # Get all users
        users = list(users_collection.find({}, {'_id': 1, 'walletAddress': 1}))
        
        if not users:
            return jsonify({'message': 'No users found'}), 200
        
        # Create batch job
        job_id = f"rec_batch_{int(time.time())}"
        batch_job = {
            'jobId': job_id,
            'jobType': 'recommendation_generation',
            'status': 'running',
            'startTime': datetime.now(),
            'totalUsers': len(users),
            'processedCount': 0,
            'failedCount': 0,
            'batchSize': 10,
            'delayBetweenBatches': 2000,  # 2 seconds
            'processedUsers': [],
            'failedUsers': []
        }
        
        batch_jobs_collection.insert_one(batch_job)
        
        # Process recommendations in batches
        processed = 0
        failed = 0
        
        logger.info(f"Starting batch generation for {len(users)} users")
        
        for i in range(0, len(users), batch_job['batchSize']):
            batch_users = users[i:i + batch_job['batchSize']]
            
            for user in batch_users:
                try:
                    user_id = str(user['_id'])
                    
                    # Generate recommendations
                    start_time = time.time()
                    recommendations = rec_engine.generate_recommendations_for_user(user_id)
                    user_profile = rec_engine.get_user_profile(user_id)
                    processing_time = (time.time() - start_time) * 1000  # in milliseconds
                    
                    # Save recommendations
                    recommendations_collection.update_one(
                        {'userId': user['_id']},
                        {
                            '$set': {
                                'userId': user['_id'],
                                'walletAddress': user['walletAddress'],
                                'recommendations': recommendations,
                                'userProfile': user_profile,
                                'lastCalculated': datetime.now(),
                                'expiresAt': datetime.now() + timedelta(hours=6),
                                'batchId': job_id,
                                'processingTime': processing_time
                            }
                        },
                        upsert=True
                    )
                    
                    processed += 1
                    logger.info(f"Processed user {user['walletAddress']} ({processed}/{len(users)})")
                    
                except Exception as e:
                    failed += 1
                    logger.error(f"Failed to process user {user['_id']}: {e}")
                    
                    # Record failure
                    batch_jobs_collection.update_one(
                        {'jobId': job_id},
                        {
                            '$push': {'failedUsers': {'userId': str(user['_id']), 'error': str(e)}},
                            '$set': {'failedCount': failed}
                        }
                    )
            
            # Update progress
            batch_jobs_collection.update_one(
                {'jobId': job_id},
                {'$set': {'processedCount': processed}}
            )
            
            # Delay between batches
            if i + batch_job['batchSize'] < len(users):
                time.sleep(batch_job['delayBetweenBatches'] / 1000.0)
        
        # Mark job as completed
        batch_jobs_collection.update_one(
            {'jobId': job_id},
            {
                '$set': {
                    'status': 'completed',
                    'endTime': datetime.now(),
                    'processedCount': processed,
                    'failedCount': failed
                }
            }
        )
        
        logger.info(f"Batch job {job_id} completed. Processed: {processed}, Failed: {failed}")
        
        return jsonify({
            'message': 'Batch generation completed',
            'jobId': job_id,
            'totalUsers': len(users),
            'processedCount': processed,
            'failedCount': failed
        })
        
    except Exception as e:
        logger.error(f"Error in batch generation: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch/status/<job_id>', methods=['GET'])
def get_batch_status(job_id):
    """Get the status of a batch job"""
    try:
        job = batch_jobs_collection.find_one({'jobId': job_id})
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Convert ObjectIds to strings for JSON serialization
        job['_id'] = str(job['_id'])
        if 'processedUsers' in job:
            job['processedUsers'] = [str(uid) for uid in job['processedUsers']]
        
        return jsonify(job)
        
    except Exception as e:
        logger.error(f"Error getting batch status: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Scheduled job to refresh recommendations periodically
def scheduled_batch_recommendations():
    """Scheduled function to generate recommendations for all users"""
    try:
        logger.info("Starting scheduled batch recommendation generation")
        
        # Make internal request to batch generate endpoint
        with app.test_request_context():
            result = batch_generate_all_recommendations()
            logger.info(f"Scheduled batch job result: {result}")
            
    except Exception as e:
        logger.error(f"Scheduled batch job failed: {e}")

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=scheduled_batch_recommendations,
    trigger="interval",
    hours=6,  # Run every 6 hours
    id='batch_recommendations',
    replace_existing=True
)

# Start scheduler only if not in debug mode to avoid duplicate jobs
if not app.debug:
    scheduler.start()

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown() if scheduler.running else None)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)