import os


def usun_nadmiar_zdjec(folder_zdjec, folder_etykiet):
    # Pobierz listę plików z folderu ze zdjeciami
    zdjecia = [plik.split('.')[0] for plik in os.listdir(folder_zdjec)]

    # Pobierz listę plików z folderu z etykietami
    etykiety = [plik.split('.')[0] for plik in os.listdir(folder_etykiet)]

    # Znajdź nadmiarowe zdjecia
    nadmiarowe_zdjecia = set(zdjecia) - set(etykiety)

    # Usuń nadmiarowe zdjecia
    for zdjecie in nadmiarowe_zdjecia:
        # Zakładam, że są to pliki JPG, dostosuj do swojego formatu
        sciezka_do_zdjecia = os.path.join(folder_zdjec, zdjecie + '.png')
        os.remove(sciezka_do_zdjecia)
        print(f'Usunięto: {sciezka_do_zdjecia}')


# Podaj ścieżki do folderów z zdjeciami i etykietami
folder_zdjec = 'testowe_zdjecia'
folder_etykiet = 'etykiety_testowe'

# Wywołaj funkcję usuwającą nadmiarowe zdjecia
usun_nadmiar_zdjec(folder_zdjec, folder_etykiet)
