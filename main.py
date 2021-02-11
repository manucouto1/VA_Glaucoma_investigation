from datatool import DataTool
import visualize as vi

import loss as l
import chain as ch

def main():
    tool = DataTool("./Material Glaucoma/refuge_images")
    tool.loadData(r'[n|g]\d{4}[^_]')

    """
    handmade_p_tile_method, sobel_watershed_method,  otsu_local_method, funcionan decente con disc
    """

    if(1>2):
        met_list = []
        #ch.handmade_p_tile_method, ch.sobel_watershed_method, ch.otsu_local_method
        for alg in [ch.sobel_watershed_method]:
            masks = l.run_all(tool,alg, op="cup")
            ss = (alg,l.metrics(masks, l.sensitivity),  l.metrics(masks, l.specificity))
            jd = (alg,l.metrics(masks, l.DICE), l.metrics(masks, l.jaccard))
            met_list.append(ss)
            met_list.append(jd)

        vi.plot_scatter(met_list,3,2,"metrics")
        vi.show()

    else:
        # tool_entry = tool.data["g0406.png"]
        tool_entry = tool.data["n0012.png"]
        # tool_entry = tool.data["n0012.png"]
        # tool_entry = tool.data["n0005.png"]
        # tool_entry = tool.data["n0013.png"]

        
        img = tool_entry["img"]
        (_,cut, _) = ch.sobel_watershed_method(img,test=True)
        # ch.snakes(cut, op="cup", test=True)
        vi.show()
  
if __name__ == "__main__":
    main()