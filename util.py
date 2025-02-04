import numpy as np
import quaternion
import plotly.graph_objects as go

def generate_trajectory_from_delta_pos(init_p,y_delta_p):
    cur_p = np.array(init_p)
    pred_p = []
    pred_p.append(np.array(cur_p))

    for delta_p in y_delta_p:
        cur_p = cur_p + delta_p
        pred_p.append(np.array(cur_p))

    return np.reshape(pred_p, (len(pred_p), 3))

def generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q):
    cur_p = np.array(init_p)
    cur_q = quaternion.from_float_array(init_q)
    pred_p = []
    pred_p.append(np.array(cur_p))

    for [delta_p, delta_q] in zip(y_delta_p, y_delta_q):
        cur_p = cur_p + np.matmul(quaternion.as_rotation_matrix(cur_q), delta_p.T).T
        cur_q = cur_q * quaternion.from_float_array(delta_q).normalized()
        pred_p.append(np.array(cur_p))

    return np.reshape(pred_p, (len(pred_p), 3))


def generate_trajectory_3d(init_l, init_theta, init_psi, y_delta_l, y_delta_theta, y_delta_psi):
    cur_l = np.array(init_l)
    cur_theta = np.array(init_theta)
    cur_psi = np.array(init_psi)
    pred_l = []
    pred_l.append(np.array(cur_l))

    for [delta_l, delta_theta, delta_psi] in zip(y_delta_l, y_delta_theta, y_delta_psi):
        cur_theta = cur_theta + delta_theta
        cur_psi = cur_psi + delta_psi
        cur_l[0] = cur_l[0] + delta_l * np.sin(cur_theta) * np.cos(cur_psi)
        cur_l[1] = cur_l[1] + delta_l * np.sin(cur_theta) * np.sin(cur_psi)
        cur_l[2] = cur_l[2] + delta_l * np.cos(cur_theta)
        pred_l.append(np.array(cur_l))

    return np.reshape(pred_l, (len(pred_l), 3))

def plot_traj(gt_trajectory, pred_trajectory, fig_path, trajectory_key):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=pred_trajectory[:, 0],
        y=pred_trajectory[:, 1],
        z=pred_trajectory[:, 2],
        mode="lines",
        name="Predicted",
        line=dict(width=4)
    ))

    fig.add_trace(go.Scatter3d(
        x=gt_trajectory[:, 0],
        y=gt_trajectory[:, 1],
        z=gt_trajectory[:, 2],
        mode="lines",
        name="Ground Truth",
        line=dict(width=4)
    ))


    # Update the axis names and title
    fig.update_layout(
        scene=dict(
            xaxis_title=dict(text="X (m)", font=dict(family="Arial, bold", size=18)),
            yaxis_title=dict(text="Y (m)", font=dict(family="Arial, bold", size=18)),
            zaxis_title=dict(text="Z (m)", font=dict(family="Arial, bold", size=18)),
            aspectmode='cube'
        ),

        title=dict(text=trajectory_key, font=dict(size=24), x=0.5, y=0.9),
        height=600,
        width=1000,
        margin=dict(l=50, r=50, b=50, t=100),
        scene_aspectmode='cube'
    )

    # Save the plot as an HTML file
    fig.write_html(fig_path)