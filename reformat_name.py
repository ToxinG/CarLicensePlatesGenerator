import os


def main():
    path = 'images/plates_text'
    for name_jpg in os.listdir('images/plates_text'):
        name = name_jpg.split('.')[0]
        text = name[:9]
        base_name = name[10:]
        new_name_jpg = base_name + '_' + text + '.jpg'
        os.rename(os.path.join(path, name_jpg), os.path.join(path, new_name_jpg))


if __name__ == '__main__':
    main()
