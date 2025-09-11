from urllib.request import urlretrieve
import vk, os, time, math
import datetime
import time
import re

vkapi = vk.API(access_token='...', v='5.131')

group_url = "https://vk.com/petretrieval"

screen_name = group_url.split('/')[-1]

response = vkapi.utils.resolveScreenName(screen_name=screen_name)

if response.get('type') == 'group':
    id = response['object_id']
else:
    print("Ошибка: Это не является сообществом.")
    exit()

global_dict = {}
urls = []  

def search_posts(group_id):
    global global_dict
    global urls

    if not os.path.exists('photos'):
        os.makedirs('photos')

    if str(group_id) in global_dict:
        print("Уже был")
        return
    global_dict[str(group_id)] = 1
    print("Словарь = ", global_dict)

    start_date = datetime.datetime(2025, 4, 10)  
    #end_date = datetime.datetime(2024, 3, 31)
    end_date = datetime.datetime.now()  

    start_time = int(start_date.timestamp())
    end_time = int(end_date.timestamp())

    execute_script = """
    var responses = [],
        i = 0;
    while (i < 2) {
      var posts = API.wall.get({
        "owner_id": -%s,
        "offset": i * 100,
        "count": 100
      }).items;
      responses = responses + posts;
      i = i + 1;
    }
    return responses;
    """ % (group_id)

    posts = vkapi.execute(code=execute_script)
    #posts = vkapi.wall.get(owner_id=-int(group_id), count=20, start_time=start_time, end_time=end_time)

    size = 0
    break_size = 0

    for post in posts:
        post_date_unix = post['date']
        if not(start_time <= post_date_unix <= end_time):
            break_size += 1
            if break_size == 2:
                break
        size += 1
        if 'copy_history' in post:
            for copy_post in post['copy_history']:
                if copy_post['owner_id'] < 0:
                    owner_id = -copy_post['owner_id']
                    print(f"Пост был переслан из сообщества с id {owner_id}.")
                    search_posts(owner_id)
                else:
                    print("Пост был переслан из профиля пользователя.")
        # text = post.get('text', 'No text')
        # print(f"Текст: {text}")
        post_id = post['id']

        attachments = post.get('attachments', [])
        for attachment in attachments:
            if attachment['type'] == 'photo':
                photo = attachment['photo']
                photo_url = photo['sizes'][-1]['url']
                photo_id = photo['id']
                photo_extension = photo_url.split('.')[-1]
                photo_filename = f"photos/post_{post_id}_photo_{photo_id}.{photo_extension}"
                new_filename = re.sub(r'\?.*$', '', photo_filename)
                urlretrieve(photo_url, new_filename)
                urls.append(photo_url)  

        comments = vkapi.wall.getComments(owner_id=-int(group_id), post_id=post_id, count=100)

        for comment in comments['items']:
            attachments = comment.get('attachments', [])
            for attachment in attachments:
                if attachment['type'] == 'photo':
                    photo = attachment['photo']
                    photo_url = photo['sizes'][-1]['url']
                    photo_id = photo['id']
                    photo_extension = photo_url.split('.')[-1]
                    photo_filename = f"photos/post_{post_id}_comment_{comment['id']}_photo_{photo_id}.{photo_extension}"
                    new_filename = re.sub(r'\?.*$', '', photo_filename)
                    urlretrieve(photo_url, new_filename)
                    urls.append(photo_url)  

        time.sleep(0.5) 

    print("Фотографии успешно загружены в папку 'photos'.")
    print(size)

search_posts(id)