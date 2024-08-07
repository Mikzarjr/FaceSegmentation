class FaceSeg:
    def __init__(self, image_path: str):
        """
        TODO: No need do save any masks until the final masks got

        :param image_path: Path to image desired for segmentation
        :return: Combined mask in COMBINED_MASK_DIR, All masks for each class separately in SPLIT_MASK_DIR
        """
        self.image_path = image_path

    def SegmentImage(self, qwe: list, asd: str) -> list:
        """
        TODO: 1 check mask type
        TODO: return dict[str:class, list[int]:mask]

        :param asd: asd asd asd asd
        :type asd: str
        :param qwe: qwe qwe qwe qwe qwe qwe
        :type qwe: list
        :rtype: list
        :return: qwe which is qwe
        """

        qwe.append([self.image_path, asd])
        return qwe


S = FaceSeg("qwe")

S.SegmentImage.__doc__
