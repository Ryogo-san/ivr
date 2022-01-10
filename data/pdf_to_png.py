import os

<<<<<<< HEAD

<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
=======
>>>>>>> origin/master
import pdf2image


def get_pdf_list(data_dir):
    """get pdf file list

    get pdf file list in data_dir

    Args:
        data_dir: data folder

    Returns:
        pdf_list: the list of the pdf file names in data_dir
    """
<<<<<<< HEAD
<<<<<<< Updated upstream
=======
>>>>>>> origin/master
    pdf_list=[]
    for dirname,_,filenames in os.walk(data_dir):
        for filename in filenames:
            pdf_file=os.path.join(dirname,filename)
            if filename[-4:]==".pdf":
<<<<<<< HEAD
=======
    pdf_list = []
    for dirname, _, filenames in os.walk(data_dir):
        for filename in filenames:
            pdf_file = os.path.join(dirname, filename)
            if filename[-4:] == ".pdf":
>>>>>>> Stashed changes
=======
>>>>>>> origin/master
                pdf_list.append(pdf_file)

    return pdf_list


def pdf_to_image(pdf_list):
    """convert pdf to image

    convert pdf files in pdf_list to png images

    Args:
        pdf_list: pdf file names list

    Returns:
        None
    """
    for pdf_file in pdf_list:
<<<<<<< HEAD
<<<<<<< Updated upstream
=======
>>>>>>> origin/master
        file_name=pdf_file[:-4]
        images=pdf2image.convert_from_path(pdf_file)
        for idx,image in enumerate(images):
            image.save(file_name+f"{idx+1}"+".png","png")


if __name__=="__main__":
<<<<<<< HEAD
=======
        file_name = pdf_file[:-4]
        images = pdf2image.convert_from_path(pdf_file)
        for idx, image in enumerate(images):
            image.save(file_name + f"{idx+1}" + ".png", "png")


if __name__ == "__main__":
>>>>>>> Stashed changes
=======
>>>>>>> origin/master
    pdf_to_image(get_pdf_list("./original_input"))
