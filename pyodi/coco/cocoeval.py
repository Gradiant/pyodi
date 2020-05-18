# flake8: noqa

import datetime
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

Num = Union[int, float]


class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    def __init__(
        self,
        ground_truth: COCO,
        predictions: COCO,
        area_ranges: Optional[Dict[str, Tuple[Num, Num]]] = None,
        iou_thresholds: Optional[List[float]] = None,
        recall_thresholds: Optional[List[float]] = None,
        score_thresholds: Optional[List[float]] = None,
        max_detections: Tuple[int, ...] = (1, 10, 100),
        lrp_iou_threshold: float = 0.5,
        f1_iou_threshold: float = 0.5,
    ):
        """

        """
        if area_ranges is None:
            area_ranges = {
                "all": (0 ** 2, 1e5 ** 2),
                "small": (0 ** 2, 32 ** 2),
                "medium": (32 ** 2, 96 ** 2),
                "large": (96 ** 2, 1e5 ** 2),
            }

        if iou_thresholds is None:
            iou_thresholds = np.arange(0.5, 1, 0.05)

        if recall_thresholds is None:
            recall_thresholds = np.arange(0, 1.01, 0.01)

        if score_thresholds is None:
            score_thresholds = np.arange(0, 1.01, 0.01)

        self.ground_truth = ground_truth
        self.predictions = predictions

        self.area_ranges = area_ranges
        self.iou_thresholds = iou_thresholds
        self.recall_thresholds = recall_thresholds
        self.score_thresholds = score_thresholds
        self.max_detections = sorted(max_detections)
        self.lrp_iou_threshold = lrp_iou_threshold
        self.f1_iou_threshold = f1_iou_threshold

        #  per-image per-category evaluation results [KxAxI] elements
        self.evalImgs: Dict[Any, Any] = defaultdict(list)
        self.eval: Dict[Any, Any] = {}  # accumulated evaluation results
        self._gts: Dict[Any, Any] = defaultdict(list)  # gt for evaluation
        self._dts: Dict[Any, Any] = defaultdict(list)  # dt for evaluation

        self.ious: Dict[Any, Any] = {}

        self.image_ids = list(np.unique(sorted(ground_truth.getImgIds())))
        self.category_ids = list(np.unique(sorted(ground_truth.getCatIds())))
        self.maxDets = [1, 10, 100]

    def _prepare(self):
        """Prepare ._gts and ._dts for evaluation based on params."""
        gts = self.ground_truth.loadAnns(
            self.ground_truth.getAnnIds(imgIds=self.image_ids, catIds=self.category_ids)
        )
        dts = self.predictions.loadAnns(
            self.predictions.getAnnIds(imgIds=self.image_ids, catIds=self.category_ids)
        )

        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]

        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)

    def evaluate(self):
        """Run per image evaluation on given images and store results.

        The results are saved in self.evalImgs (list of dict).

        """
        tic = time.time()
        print("Running per image evaluation...")
        self._prepare()

        self.ious = {
            (image_id, category_id): self.compute_iou(image_id, category_id)
            for image_id in self.image_ids
            for category_id in self.category_ids
        }

        self.eval_images = [
            self.evaluateImg(image_id, category_id, area_range, self.max_detections[-1])
            for category_id in self.category_ids
            for area_range in self.area_ranges.values()
            for image_id in self.image_ids
        ]
        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def compute_iou(self, image_id: int, category_id: int):
        """Computes the IoU value between ground truth and predictions.

        Args:
            image_id: Image Id.
            category_id: Category Id.

        Returns:
            IoU values between ground truth and predictions.

        """
        gt = self._gts[image_id, category_id]
        dt = self._dts[image_id, category_id]

        if not (gt or gt):
            return []

        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]

        if len(dt) > self.max_detections[-1]:
            dt = dt[0 : self.max_detections[-1]]

        g = [g["bbox"] for g in gt]
        d = [d["bbox"] for d in dt]

        # compute iou between each dt and gt region
        iscrowd = [int(o["iscrowd"]) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def evaluateImg(self, image_id, category_id, area_range, maxDet):
        """Perform evaluation for single category and image.

        Args:
            image_id: Image Id.
            category_id: Category Id.
            area_range: Area range.
            maxDet: Maximum number of detections.

        Returns:
            Dict with single image results.

        """
        ground_truth = self._gts[image_id, category_id]
        predictions = self._dts[image_id, category_id]

        if not (ground_truth or predictions):
            return None

        for g in ground_truth:
            if g["ignore"] or (g["area"] < area_range[0] or g["area"] > area_range[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in ground_truth], kind="mergesort")
        ground_truth = [ground_truth[i] for i in gtind]

        dtind = np.argsort([-d["score"] for d in predictions], kind="mergesort")
        predictions = [predictions[i] for i in dtind[:maxDet]]

        iscrowd = [int(o["iscrowd"]) for o in ground_truth]

        ious = (
            self.ious[image_id, category_id][:, gtind]
            if len(self.ious[image_id, category_id]) > 0
            else self.ious[image_id, category_id]
        )

        n_thresholds = len(self.iou_thresholds)
        n_ground_truth = len(ground_truth)
        n_predictions = len(predictions)

        ground_truth_matched = np.zeros((n_thresholds, n_ground_truth))
        predictions_matched = np.zeros((n_thresholds, n_predictions))

        predictions_iou = np.zeros(n_predictions)

        ground_truth_ignore = np.array([g["_ignore"] for g in ground_truth])
        predictions_ignore = np.zeros((n_thresholds, n_predictions))

        if len(ious) > 0:
            for tind, t in enumerate(self.iou_thresholds):
                for dind, d in enumerate(predictions):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    match = -1
                    for gind, g in enumerate(ground_truth):
                        # if this gt already matched, and not a crowd, continue
                        if ground_truth_matched[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if (
                            match > -1
                            and ground_truth_ignore[match] == 0
                            and ground_truth_ignore[gind] == 1
                        ):
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        match = gind
                    if match == -1:
                        continue
                    # if match made store id of match for both dt and gt
                    predictions_ignore[tind, dind] = ground_truth_ignore[match]
                    predictions_matched[tind, dind] = ground_truth[match]["id"]
                    ground_truth_matched[tind, match] = d["id"]

                    # LRP
                    if t == self.lrp_iou_threshold:
                        predictions_iou[dind] = iou

        # set unmatched detections outside of area range to ignore
        a = np.array(
            [
                d["area"] < area_range[0] or d["area"] > area_range[1]
                for d in predictions
            ]
        ).reshape((1, n_predictions))
        predictions_ignore = np.logical_or(
            predictions_ignore,
            np.logical_and(predictions_matched == 0, np.repeat(a, n_thresholds, 0)),
        )
        # store results for given image and category
        return {
            "image_id": image_id,
            "category_id": category_id,
            "area_range": area_range,
            "max_detections": self.max_detections[-1],
            "dtIds": [d["id"] for d in predictions],
            "gtIds": [g["id"] for g in ground_truth],
            "dtMatches": predictions_matched,
            "gtMatches": ground_truth_matched,
            "dtScores": [d["score"] for d in predictions],
            "gtIgnore": ground_truth_ignore,
            "dtIgnore": predictions_ignore,
            "dtIoUs": predictions_iou,
        }

    def accumulate(self):
        """Accumulate per image evaluation results and store the result in self.eval."""
        print("Accumulating evaluation results...")
        tic = time.time()

        n_iou_thresholds = len(self.iou_thresholds)
        n_recall_thresholds = len(self.recall_thresholds)
        n_categories = len(self.category_ids)
        n_max_detections = len(self.max_detections)
        n_images = len(self.image_ids)
        n_area_ranges = len(self.area_ranges)
        n_score_thresholds = len(self.score_thresholds)

        # -1 for the precision of absent categories
        precision = -np.ones(
            (
                n_iou_thresholds,
                n_recall_thresholds,
                n_categories,
                n_area_ranges,
                n_max_detections,
            )
        )
        recall = -np.ones(
            (n_iou_thresholds, n_categories, n_area_ranges, n_max_detections)
        )
        scores = -np.ones(
            (
                n_iou_thresholds,
                n_recall_thresholds,
                n_categories,
                n_area_ranges,
                n_max_detections,
            )
        )

        omega = np.zeros((n_score_thresholds, n_categories))
        nhat = np.zeros((n_score_thresholds, n_categories))
        mhat = np.zeros((n_score_thresholds, n_categories))
        LRPError = -np.ones((n_score_thresholds, n_categories))
        LocError = -np.ones((n_score_thresholds, n_categories))
        FPError = -np.ones((n_score_thresholds, n_categories))
        FNError = -np.ones((n_score_thresholds, n_categories))
        OptLRPError = -np.ones((1, n_categories))
        OptLocError = -np.ones((1, n_categories))
        OptFPError = -np.ones((1, n_categories))
        OptFNError = -np.ones((1, n_categories))
        Threshold = -np.ones((1, n_categories))

        # retrieve E at each category, area range, and max number of detections
        for k in range(n_categories):
            Nk = k * n_area_ranges * n_images
            for a, area_range in enumerate(self.area_ranges):
                Na = a * n_images
                for m, maxDet in enumerate(self.max_detections):
                    E = [self.eval_images[Nk + Na + i] for i in range(n_images)]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e["dtScores"][:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e["dtMatches"][:, :maxDet] for e in E], axis=1
                    )[:, inds]
                    dtIg = np.concatenate(
                        [e["dtIgnore"][:, :maxDet] for e in E], axis=1
                    )[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue

                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                    # COCO mAP
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((n_recall_thresholds,))
                        ss = np.zeros((n_recall_thresholds,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        recall_inds = np.searchsorted(
                            rc, self.recall_thresholds, side="left"
                        )
                        try:
                            for ri, pi in enumerate(recall_inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)

                    # LRP
                    if area_range == "all" and maxDet == max(self.max_detections):
                        IoUoverlap = np.concatenate([e["dtIoUs"][:maxDet] for e in E])[
                            inds
                        ]

                        for i in range(len(IoUoverlap)):
                            if IoUoverlap[i] != 0:
                                IoUoverlap[i] = 1 - IoUoverlap[i]

                        # Only use tps for lrp_iou_threshold threshold
                        tps = tps[self.iou_thresholds == self.lrp_iou_threshold]
                        fps = fps[self.iou_thresholds == self.lrp_iou_threshold]
                        IoUoverlap = np.multiply(IoUoverlap, tps)

                        for s, s0 in enumerate(self.score_thresholds):
                            thrind = np.sum(dtScoresSorted >= s0)
                            omega[s, k] = np.sum(tps[:thrind])
                            nhat[s, k] = np.sum(fps[:thrind])
                            mhat[s, k] = npig - omega[s, k]
                            normalize = np.maximum((omega[s, k] + nhat[s, k]), npig)
                            FPError[s, k] = (1 - self.lrp_iou_threshold) * (
                                nhat[s, k] / normalize
                            )
                            FNError[s, k] = (1 - self.lrp_iou_threshold) * (
                                mhat[s, k] / normalize
                            )
                            Z = (omega[s, k] + mhat[s, k] + nhat[s, k]) / normalize
                            LRPError[s, k] = (
                                (np.sum(IoUoverlap[:thrind]) / normalize)
                                + FPError[s, k]
                                + FNError[s, k]
                            )
                            LRPError[s, k] = LRPError[s, k] / Z
                            LRPError[s, k] = LRPError[s, k] / (
                                1 - self.lrp_iou_threshold
                            )
                            LocError[s, k] = np.sum(IoUoverlap[:thrind]) / omega[s, k]
                            FPError[s, k] = nhat[s, k] / (omega[s, k] + nhat[s, k])
                            FNError[s, k] = mhat[s, k] / npig

                        OptLRPError[0, k] = min(LRPError[:, k])
                        ind = np.argmin(LRPError[:, k])
                        OptLocError[0, k] = LocError[ind, k]
                        OptFPError[0, k] = FPError[ind, k]
                        OptFNError[0, k] = FNError[ind, k]
                        Threshold[0, k] = ind * 0.01

        moLRPLoc = np.nanmean(OptLocError)
        moLRPFP = np.nanmean(OptFPError)
        moLRPFN = np.nanmean(OptFNError)
        moLRP = np.mean(OptLRPError)

        precision_for_f1 = precision[self.iou_thresholds == self.f1_iou_threshold].mean(
            0
        )
        recall_for_f1 = recall[self.iou_thresholds == self.f1_iou_threshold]
        f1 = 2 * (
            (precision_for_f1 * recall_for_f1) / (precision_for_f1 + recall_for_f1)
        )
        self.eval = {
            "counts": [
                n_iou_thresholds,
                n_recall_thresholds,
                n_categories,
                n_area_ranges,
                n_max_detections,
                n_score_thresholds,
            ],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "scores": scores,
            "LRPError": LRPError,
            "BoxLocComp": LocError,
            "FPComp": FPError,
            "FNComp": FNError,
            "oLRPError": OptLRPError,
            "oBoxLocComp": OptLocError,
            "oFPComp": OptFPError,
            "oFNComp": OptFNError,
            "moLRP": moLRP,
            "moLRPLoc": moLRPLoc,
            "moLRPFP": moLRPFP,
            "moLRPFN": moLRPFN,
            "OptThresholds": Threshold,
        }
        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def summarize(self):
        """Compute and display summary metrics for evaluation results.

        Note this function can *only* be applied on the default parameter setting.

        """

        def _summarize(
            mode="precision",
            iouThr=None,
            areaRng="all",
            maxDets=100,
            per_category=False,
        ):
            iStr = " {:<40} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = f"Average {mode.title()}"
            typeStr = {"precision": "(AP)", "recall": "(AR)", "f1": "(F1)"}[mode]
            iouStr = (
                "{:0.2f}:{:0.2f}".format(
                    self.iou_thresholds[0], self.iou_thresholds[-1]
                )
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = list(self.area_ranges.keys()).index(areaRng)
            mind = self.max_detections.index(maxDets)

            results = self.eval[mode]

            if mode in ["precision", "recall", "f1"]:
                if iouThr is not None:
                    results = results[np.where(iouThr == self.iou_thresholds)[0]]
                results = results[..., aind, mind]

            if per_category:
                per_category_results = results.mean(tuple(range(results.ndim - 1)))

                """
                for cat_index, cat_id in enumerate(self.category_ids):
                    class_name = self.ground_truth.cats[cat_id]["name"]
                    class_titleStr = f"{titleStr}: {class_name}"
                    print(
                        iStr.format(
                            class_titleStr,
                            typeStr,
                            iouStr,
                            areaRng,
                            maxDets,
                            per_category_results[cat_index],
                        )
                    )
                """
                return per_category_results
            else:
                if np.sum(results > -1) == 0:
                    mean_results = -1
                else:
                    mean_results = np.mean(results[results > -1])
                print(
                    iStr.format(
                        titleStr, typeStr, iouStr, areaRng, maxDets, mean_results
                    )
                )
                return mean_results

        self.mean_stats = np.zeros((13,))

        print("\nCOCO mAP\n")
        print("------------------------------------------------------\n")
        self.mean_stats[0] = _summarize("precision")
        self.mean_stats[1] = _summarize(
            "precision", iouThr=0.5, maxDets=self.max_detections[2]
        )
        self.mean_stats[2] = _summarize(
            "precision", iouThr=0.75, maxDets=self.max_detections[2]
        )
        self.mean_stats[3] = _summarize(
            "precision", areaRng="small", maxDets=self.max_detections[2]
        )
        self.mean_stats[4] = _summarize(
            "precision", areaRng="medium", maxDets=self.max_detections[2]
        )
        self.mean_stats[5] = _summarize(
            "precision", areaRng="large", maxDets=self.max_detections[2]
        )

        self.mean_stats[6] = _summarize("recall", maxDets=self.max_detections[0])
        self.mean_stats[7] = _summarize("recall", maxDets=self.max_detections[1])
        self.mean_stats[8] = _summarize("recall", maxDets=self.max_detections[2])
        self.mean_stats[9] = _summarize(
            "recall", areaRng="small", maxDets=self.max_detections[2]
        )
        self.mean_stats[10] = _summarize(
            "recall", areaRng="medium", maxDets=self.max_detections[2]
        )
        self.mean_stats[11] = _summarize(
            "recall", areaRng="large", maxDets=self.max_detections[2]
        )
        self.mean_stats[12] = _summarize("f1", iouThr=0.5)

        print("\nMean Optimal LRP and Components\n")
        print("------------------------------------------------------\n")
        print(
            "moLRP={:0.4f}, moLRP_LocComp={:0.4f}, moLRP_FPComp={:0.4f}, moLRP_FPComp={:0.4f} \n".format(
                self.eval["moLRP"],
                self.eval["moLRPLoc"],
                self.eval["moLRPFP"],
                self.eval["moLRPFN"],
            )
        )

        self.per_category_stats = np.zeros((12, len(self.category_ids)))
        self.per_category_stats[0] = _summarize("precision", per_category=True)
        self.per_category_stats[1] = _summarize(
            "recall", iouThr=0.5, maxDets=self.max_detections[2], per_category=True
        )
        self.per_category_stats[2] = _summarize(
            "precision", iouThr=0.75, maxDets=self.max_detections[2], per_category=True
        )
        self.per_category_stats[3] = _summarize(
            "precision",
            areaRng="small",
            maxDets=self.max_detections[2],
            per_category=True,
        )
        self.per_category_stats[4] = _summarize(
            "precision",
            areaRng="medium",
            maxDets=self.max_detections[2],
            per_category=True,
        )
        self.per_category_stats[5] = _summarize(
            "precision",
            areaRng="large",
            maxDets=self.max_detections[2],
            per_category=True,
        )

    def __str__(self):
        self.summarize()
