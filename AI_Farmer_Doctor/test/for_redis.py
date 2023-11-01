import redis  # redis的基本配置

REDIS_CONFIG = {
    "host": "127.0.0.1",
    "port": 6379
}
link_pool = redis.ConnectionPool(**REDIS_CONFIG)
con_redis = redis.StrictRedis(connection_pool=link_pool)  # my_ id是一个用户的标识
my_id = "0001"
last_sentence = "这是最后一句:".encode('utf-8')  # value是需要记录的数据具体内容
my_output = "输出Redis".encode('utf-8')
con_redis.hset(my_id, last_sentence, my_output)
print(con_redis.hget(my_id, last_sentence).decode('utf-8'))
