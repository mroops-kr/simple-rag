'''

    mysql_util.py - 테스트 파일

'''
from dotenv import load_dotenv
load_dotenv()

from mysql_util import MysqlClient

# 단건 조회
def test_select_one():
    with MysqlClient() as client:
        result = client.auto.selectOne('actor', {
            'actorId': 1,
        })
        print(result)

# 목록 조회
def test_select_list():
    with MysqlClient() as client:
        results = client.auto.selectList('actor', {
            'actorId': [1, 2, 3], # >> where actor_id in (1, 2, 3)
            'firstName': '' # ignored, empty string
        })

        for result in results:
            print(result)

# 건수 조회
def test_select_count():
    with MysqlClient() as client:
        result = client.auto.count('actor', {
            'actorId': [1, 2, 3], # >> where actor_id in (1, 2, 3)
        })
        print(result)

# 페이징 조회
def test_select_pagination():
    with MysqlClient() as client:
        # 건수 조회
        count = client.auto.count('actor', {
            'order_by': 'actor_id desc',
            'offset': 10,
            'limit': 10
        })
        print(count)
        # 페이징 데이터 조회
        results = client.auto.selectList('actor', {
            'order_by': 'actor_id desc',
            'offset': 10,
            'limit': 10
        })
        for result in results:
            print(result)

# INSERT
def test_insert():
    with MysqlClient() as client:
        # sequence
        # actorId = client.auto.nextVal('actor')

        client.auto.insert('actor', {
                'actorId': 301,
                'firstName': 'Heung-Min',
                'lastName': 'Son',
                'lastUpdate': '2024-01-01'
            })
        client.commit()

# UPDATE
def test_update():
    with MysqlClient() as client:
        client.auto.update('actor', {
            'actorId': 301, # Key
            'lastUpdate': '2024-12-25'
        })
        client.commit()

# DELETE
def test_delete():
    with MysqlClient() as client:
        client.auto.delete('actor', {
            'actorId': 301, # Key
        })
        client.commit()

# SEQUENCE
def test_sequence():
    with MysqlClient() as client:
        seq = client.auto.nextVal('actor')
        print(seq)

# sql 기반 - 기존 방식 - auto 제거
def test_asis():
    with MysqlClient() as client:
        results = client.selectList(
            sql = "SELECT * FROM actor WHERE actor_id >= %s ORDER BY actor_id LIMIT %s OFFSET %s",
            param = (5, 10, 0)
        )
        for result in results:
            print(result)


# 단건 조회
# test_select_one()

# 목록 조회
test_select_list()

# 건수 조회
# test_select_count()

# 페이징 조회
# test_select_pagination()

# INSERT
# test_insert()

# UPDATE
# test_update()

# DELETE
# test_delete()

# SEQUENCE
# test_sequence()

# sql 기반 - 기존 방식
# test_asis()

#           python utils/mysql_test.py

'''
테스트 테이블

CREATE TABLE IF NOT EXISTS `actor` (
    `actor_id` smallint unsigned NOT NULL AUTO_INCREMENT,
    `first_name` varchar(45) NOT NULL,
    `last_name` varchar(45) NOT NULL,
    `last_update` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`actor_id`),
    KEY `idx_actor_last_name` (`last_name`)
) ENGINE=InnoDB AUTO_INCREMENT=302 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

INSERT INTO `actor` (`actor_id`, `first_name`, `last_name`, `last_update`) VALUES
	(1, 'PENELOPE', 'GUINESS', '2006-02-14 19:34:33'),
	(2, 'NICK', 'WAHLBERG', '2006-02-14 19:34:33'),
	(3, 'ED', 'CHASE', '2006-02-14 19:34:33'),
	(4, 'JENNIFER', 'DAVIS', '2006-02-14 19:34:33'),
	(5, 'JOHNNY', 'LOLLOBRIGIDA', '2006-02-14 19:34:33'),
	(6, 'BETTE', 'NICHOLSON', '2006-02-14 19:34:33'),
	(7, 'GRACE', 'MOSTEL', '2006-02-14 19:34:33'),
	(8, 'MATTHEW', 'JOHANSSON', '2006-02-14 19:34:33'),
	(9, 'JOE', 'SWANK', '2006-02-14 19:34:33'),
	(10, 'CHRISTIAN', 'GABLE', '2006-02-14 19:34:33'),
	(11, 'ZERO', 'CAGE', '2006-02-14 19:34:33'),
	(12, 'KARL', 'BERRY', '2006-02-14 19:34:33'),
	(13, 'UMA', 'WOOD', '2006-02-14 19:34:33'),
	(14, 'VIVIEN', 'BERGEN', '2006-02-14 19:34:33'),
	(15, 'CUBA', 'OLIVIER', '2006-02-14 19:34:33'),
	(16, 'FRED', 'COSTNER', '2006-02-14 19:34:33'),
	(17, 'HELEN', 'VOIGHT', '2006-02-14 19:34:33'),
	(18, 'DAN', 'TORN', '2006-02-14 19:34:33'),
	(19, 'BOB', 'FAWCETT', '2006-02-14 19:34:33'),
	(20, 'LUCILLE', 'TRACY', '2006-02-14 19:34:33'),
	(21, 'KIRSTEN', 'PALTROW', '2006-02-14 19:34:33'),
	(22, 'ELVIS', 'MARX', '2006-02-14 19:34:33'),
	(23, 'SANDRA', 'KILMER', '2006-02-14 19:34:33'),
	(24, 'CAMERON', 'STREEP', '2006-02-14 19:34:33'),
	(25, 'KEVIN', 'BLOOM', '2006-02-14 19:34:33');

CREATE TABLE IF NOT EXISTS `table_seq` (
    `table_id` varchar(45) NOT NULL,
    `seq_va` int NOT NULL,
    PRIMARY KEY (`table_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

INSERT INTO `table_seq` (`table_id`, `seq_va`) VALUES
	('actor', 200);

'''