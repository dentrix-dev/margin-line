import os
import numpy as np
import fastmesh as fm

def load_transformed_contexts_and_margins(root, transform_dir):
    cases = []
    dirpath, cases, _ = next(os.walk(root))

    def apply_transform(points, transform):
        homog = np.hstack([points, np.ones((points.shape[0], 1))])
        return (homog @ transform.T)[:, :3]

    cases_dict = {}
    for case in cases:
        context_file_path = os.path.join(dirpath, case, case[-2:] + "_margin_context.bmesh")
        margin_file_path  = os.path.join(dirpath, case, case[-2:] + "_margin.pts")

        transform_name = case + '.npy'
        transform_path = os.path.join(transform_dir, transform_name)

        transform = np.load(transform_path)  

        loaded_context = fm.load(context_file_path)[0] 
        loaded_margin  = np.loadtxt(margin_file_path, skiprows=1)
        
        transformed_context = apply_transform(loaded_context, transform)
        transformed_margin  = apply_transform(loaded_margin, transform)

        centroid = np.mean(transformed_context)
        transformed_context -= centroid
        transformed_margin  -= centroid

        cases_dict[case] = {"context": transformed_context, "margin": transformed_margin}
    return cases_dict
