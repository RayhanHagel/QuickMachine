from tkinter import filedialog
from pathlib import Path



def get_file_type(path: str) -> str:
    file_extension = Path(path).suffix.lower()
    match file_extension:
        case ".cif":
            return "cif"
        case ".xlsx" | ".xls":
            return "excel"
        case ".csv":
            return "csv"
        case ".jpg" | ".jpeg" | ".png":
            return "image"
        case _:
            return None





def choose_folder():
    while True:
        location = filedialog.askdirectory(title='Select Folder')
        
        cif_files = list(Path(location).glob('*.cif'))
        excel_files = list(Path(location).glob('*.xlsx')) + list(Path(location).glob('*.xls'))
        csv_files = list(Path(location).glob('*.csv'))
        image_files = list(Path(location).glob('*.jpg')) + list(Path(location).glob('*.jpeg')) + list(Path(location).glob('*.png'))
        
        path_files = cif_files + excel_files + csv_files + image_files
        files = [str(file) for file in path_files]
        if files != []:
            break
    return files




def choose_file():
    while True:
        files = [filedialog.askopenfilename(
            title='Select File to Parse',
            filetypes=[('CIF Files', '*.cif'),
                       ('Excel files', '*.xlsx *.xls'),
                       ('CSV files', '*.csv'),
                       ('Image files', '*.jpg *.jpeg *.png')
            ]
        )]
        if files != []:
            break
    return files




if __name__ == "__main__":
    print(choose_file())
    print(choose_folder())