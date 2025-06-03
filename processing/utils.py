import cv2


def perform_processing(cap: cv2.VideoCapture) -> dict[str, int]:
    # TODO: add video processing here
    return {
        "liczba_samochodow_osobowych_z_prawej_na_lewa": 5,
        "liczba_samochodow_osobowych_z_lewej_na_prawa": 5,
        "liczba_samochodow_ciezarowych_autobusow_z_prawej_na_lewa": 5,
        "liczba_samochodow_ciezarowych_autobusow_z_lewej_na_prawa": 5,
        "liczba_tramwajow": 5,
        "liczba_pieszych": 5,
        "liczba_rowerzystow": 5
    }