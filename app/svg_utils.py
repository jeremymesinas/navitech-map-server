def _cnt_to_path_d(cnt) -> str:
    # cnt shape: (K,1,2)
    pts = cnt.squeeze(1)  # -> (K,2)
    if len(pts) == 0:
        return ""
    cmds = [f"M {pts[0][0]} {pts[0][1]}"]
    for x, y in pts[1:]:
        cmds.append(f"L {x} {y}")
    cmds.append("Z")
    return " ".join(cmds)

def contours_to_svg(contours, width: int, height: int) -> str:
    header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    body_parts = []
    for i, cnt in enumerate(contours):
        d = _cnt_to_path_d(cnt)
        if not d:
            continue
        body_parts.append(
            f'<path d="{d}" fill="#f9dfb9" stroke="#dfb77e" stroke-width="2" />'
        )
    footer = "</svg>"
    return header + "".join(body_parts) + footer
