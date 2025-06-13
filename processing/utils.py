import cv2
from processing.objects_detection import detection


def perform_processing(cap: cv2.VideoCapture) -> dict[str, int]:
    
    # TODO: add video processing here
    detected_obj = detection(cap, show=False, debug=False)
    
    return {
        "liczba_samochodow_osobowych_z_prawej_na_lewa": detected_obj.get("osobowy_prawo_lewo", 0),
        "liczba_samochodow_osobowych_z_lewej_na_prawa": detected_obj.get("osobowy_lewo_prawo", 0),
        "liczba_samochodow_ciezarowych_autobusow_z_prawej_na_lewa": detected_obj.get("ciezarowy_prawo_lewo", 0),
        "liczba_samochodow_ciezarowych_autobusow_z_lewej_na_prawa": detected_obj.get("ciezarowy_lewo_prawo", 0),
        "liczba_tramwajow": detected_obj.get("tramwaj", 0),
        "liczba_pieszych": detected_obj.get("pieszy", 0),
        "liczba_rowerzystow": detected_obj.get("rowerzysta", 0)
    }