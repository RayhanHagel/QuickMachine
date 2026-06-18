import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import uuid



if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


if "item_group_list" not in st.session_state:
    st.session_state.item_group_list = []


def delete_keys(keys: list):
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


def check_key(key: str):
    if key in st.session_state:
        if st.session_state[key]:
            return True
    return False


def change_file_uploader_key():
    st.session_state.uploader_key += 1


@st.cache_data(show_spinner="Reading columns...")
def extract_columns(item: str, item_type: str) -> list:
    if item_type == "CSV":
        loaded_item = pd.read_csv(item, nrows=0)
    elif item_type == "Excel":
        loaded_item = pd.read_excel(item, engine="openpyxl", nrows=0)
    return loaded_item.columns.tolist()


def role_change(role: str, group_id: str):
    group_index = next((i for i, g in enumerate(st.session_state.item_group_list) if g["id"] == group_id), None)
    if group_index is None:
        return

    current_key = f"{role}_{group_id}"
    new_selections = st.session_state[current_key]
    if new_selections is None:
        return
    
    role_to_check = ["feature_column", "target_column", "ignore_column"]
    other_roles = [r for r in role_to_check if r != role]
    
    for other_role in other_roles:
        other_key = f"{other_role}_{group_id}"
        if other_key in st.session_state and st.session_state[other_key] is not None:
            updated_list = [
                item for item in st.session_state[other_key] 
                if item not in new_selections
            ]
            
            if other_role == "ignore_column":
                all_cols = set(st.session_state.item_group_list[group_index]["columns"])
                features = set(st.session_state.get(f"feature_column_{group_id}", []))
                targets = set(st.session_state.get(f"target_column_{group_id}", []))
                st.session_state[f"ignore_column_{group_id}"] = list(all_cols - features - targets)
            else:
                st.session_state[other_key] = updated_list
                
            st.session_state.item_group_list[group_index][other_role] = st.session_state[other_key]
            
    st.session_state.item_group_list[group_index][role] = new_selections


def continue_model(model_name: str):
    model_path = Path(f"./configured_data/{model_name}")
    if model_path.exists():
        return False
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(st.session_state.item_group_list, f)
    return True


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #


# Select File or Folder
st.header(":open_file_folder: Select Files", divider="gray")
st.caption("Choose input data to be processed")
st.space("xxsmall")


# Choose File Extension
input_column = st.columns(spec=2, border=True)
input_column[0].markdown(":orange[**File Extension Type**]")
input_column[0].caption("Choose the extension type of your data")
extension_options = {
    "CIF" : [".cif"],
    "Excel" : [".xlsx", ".xls"],
    "CSV" : [".csv"],
    "Image" : [".jpg", ".jpeg", ".png"]
}
input_column[0].pills(label="empty", options=extension_options.keys(), selection_mode="single", label_visibility="collapsed", key="extension_to_search")


# Choose to upload File or Folder
input_column[1].markdown(":orange[**Search Option**]")
input_column[1].caption("Search singular file or entire directory")
if not check_key("extension_to_search"):
    delete_keys(["input_type"])
else:
    input_option = ["Folder", "Files"]
    if st.session_state.extension_to_search in ["CSV", "Excel"]:
        input_option = ["Files"]
    input_column[1].pills(label="empty", options=input_option, selection_mode="single", label_visibility="collapsed", key="input_type")


# File Upload Section
directories = []
st.space("xxsmall")
if not (check_key("input_type") and check_key("extension_to_search")):
    delete_keys(["directories"])
else:
    upload_container = st.container(border=True)
    upload_container.markdown(":orange[**Upload Files**]")
    accept_multiple = True if st.session_state.extension_to_search in ["Image", "CIF"] else False
    if st.session_state.input_type == "Folder":
        upload_container.file_uploader(
            label="empty", accept_multiple_files="directory", 
            type=extension_options[st.session_state.extension_to_search], 
            label_visibility="collapsed", key=f"file_upload_{st.session_state.uploader_key}"
        )
    elif st.session_state.input_type == "Files":
        upload_container.file_uploader(
            label="empty", accept_multiple_files=accept_multiple, 
            type=extension_options[st.session_state.extension_to_search], 
            label_visibility="collapsed", key=f"file_upload_{st.session_state.uploader_key}"
        )
    else:
        st.toast(f":red[Could not determine input type]", duration="infinite", icon=":material/apps_outage:")


# Chosen Files Confirmation
if check_key(f"file_upload_{st.session_state.uploader_key}"):
    # Show Files
    directories = st.session_state[f"file_upload_{st.session_state.uploader_key}"]
    st.space("xxsmall")
    with st.expander(label=":orange[**Selected Files**]", expanded=False):
        if isinstance(directories, list):
            for directory in directories:
                st.caption(f":ledger: {directory.name}")
        else:
            st.caption(f":ledger: {directories.name}")

    # Confirm Files
    st.space("xxsmall")
    confirm_container = st.container(border=True)
    confirm_container.markdown(":green[**Confirm the Files**]")
    confirm_column = confirm_container.columns(spec=[1, 1, 3])
    
    # Add File
    if confirm_column[0].button("Add Files", use_container_width=True, type="secondary"):
        group_id = str(uuid.uuid4())
        st.session_state.item_group_list.append({
            "id": group_id,
            "item_list": directories if isinstance(directories, list) else [directories],
            "role": None,
            "feature_column" : [],
            "target_column" : [],
            "ignore_column" : [],
            "columns" : [],
            "type": st.session_state.extension_to_search
        })
        change_file_uploader_key()
        st.rerun()
    
    # Cancel and Clear
    if confirm_column[1].button("Cancel", use_container_width=True, type="secondary"):
        change_file_uploader_key()
        st.rerun()


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #


# Assign Files
st.header(":open_file_folder: Assign Files — Features & Target", divider="gray")
if not st.session_state.item_group_list:
    st.caption("Select a file to configure first")
    st.stop()
else:
    st.caption("Select the configuration of the chosen file from the list below")
st.space("xxsmall")


# File Grouping
for group_index, item_group in enumerate(st.session_state.item_group_list):
    item_amount = len(item_group["item_list"])
    group_id = item_group["id"]
    with st.expander(label=f":red[Item Group {group_index + 1} | {item_group['type']} | {item_amount} file{'s' if item_amount > 1 else ''}]", expanded=True):
        # Image and CIF Config Option
        if item_group["type"] in ["Image", "CIF"]:
            st.markdown(":orange[**File Assignment**]")
            
            group_config = st.selectbox(
                label="empty", options=["Not Used", "Feature", "Target"], 
                label_visibility="collapsed", key=f"group_config_{group_id}"
            )
            if group_config:
                st.session_state.item_group_list[group_index]["role"] = group_config
            
            # Show File List
            st.markdown(":orange[**File List**]")
            row_amount = (item_amount + 2) // 3
            for i in range(row_amount):
                row = st.columns(spec=3)
                for j in range(3):
                    idx = i * 3 + j
                    if idx < item_amount:
                        item = item_group["item_list"][idx]
                        with row[j]:
                            st.caption(item.name)
        
        # CSV and Excel Config Option
        elif item_group["type"] in ["CSV", "Excel"]:
            if not item_group["columns"]:
                try:
                    item = item_group["item_list"][0]
                    item.seek(0)
                    cols = extract_columns(item, item_group["type"])
                    st.session_state.item_group_list[group_index]["columns"] = cols
                    st.session_state.item_group_list[group_index]["ignore_column"] = cols
                except Exception as e:
                    st.error(f"Could not read columns from {item_group['item_list'][0].name}")
                    continue
            
            # Initialize keys in session state
            feat_key = f"feature_column_{group_id}"
            target_key = f"target_column_{group_id}"
            ignore_key = f"ignore_column_{group_id}"
            if feat_key not in st.session_state:
                st.session_state[feat_key] = item_group["feature_column"]
            if target_key not in st.session_state:
                st.session_state[target_key] = item_group["target_column"]
            if ignore_key not in st.session_state:
                st.session_state[ignore_key] = item_group["ignore_column"]

            # Feature Configuration
            st.markdown(":orange[**Feature Columns**]")
            st.multiselect(
                label="empty", options=item_group["columns"],
                label_visibility="collapsed", on_change=role_change, key=feat_key,
                args=("feature_column", group_id)
            )
            
            # Target Configuration
            st.markdown(":orange[**Target Column**]")
            st.multiselect(
                label="empty", options=item_group["columns"],
                label_visibility="collapsed", on_change=role_change, key=target_key,
                args=("target_column", group_id)
            )
            
            # Show Ignored Colums
            st.markdown(":orange[**Ignored Columns**]")
            st.multiselect(
                label="empty", options=item_group["columns"],
                label_visibility="collapsed", on_change=role_change, key=ignore_key,
                args=("ignore_column", group_id)
            )
            
        # Delete Button
        if st.button("Delete", use_container_width=True, type="secondary", key=f"delete_btn_{group_id}"):
            delete_keys([
                f"feature_column_{group_id}", 
                f"target_column_{group_id}", 
                f"ignore_column_{group_id}",
                f"group_config_{group_id}"
            ])
            st.session_state.item_group_list.pop(group_index)
            st.rerun()


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #


# Check if atleast One Feature AND One Target exists
st.header(":open_file_folder: Validate", divider="gray")
feature_groups = any(
    (g["type"] in ("Image", "CIF") and g.get("role") == "Feature")
    or (g["type"] in ("CSV", "Excel") and bool(g.get("feature_column")))
    for g in st.session_state.item_group_list
)
target_groups = [
    g for g in st.session_state.item_group_list
    if (g["type"] in ("Image", "CIF") and g.get("role") == "Target")
    or (g["type"] in ("CSV", "Excel") and g.get("target_column"))
]
if not (feature_groups and target_groups):
    st.caption("Configure at least a file group first, with a feature and target.")
    st.stop()
st.space("xxsmall")


# Validate Input Files
st.caption("Recheck the File Assignment Configuration before continuing.")
st.text_input(label="empty", label_visibility="collapsed", placeholder="Enter model name", key="create_model_name")
if st.button("Validate", use_container_width=True, type="primary", key="validate_data_btn"):
    if check_key("create_model_name"):
        if not continue_model(model_name=st.session_state.create_model_name):
            st.toast(body=":red[File already exists, try another filename]", duration="infinite")
        else:
            st.toast(body=":green[Saved the input data configuration]", duration="long")
            if "page_nav" in st.session_state and "config_create" in st.session_state.page_nav:
                st.switch_page(st.session_state.page_nav["config_create"])