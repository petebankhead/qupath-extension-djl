package qupath.ext.djl.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.DivergenceCheckTrainingListener;
import ai.djl.training.listener.EpochTrainingListener;
import ai.djl.training.listener.EvaluatorTrainingListener;
import ai.djl.training.listener.LoggingTrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import com.google.common.collect.Lists;
import javafx.collections.ListChangeListener;
import javafx.embed.swing.JFXPanel;
import javafx.scene.Scene;
import javafx.scene.chart.XYChart;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.djl.DjlTools;
import qupath.lib.common.GeneralTools;
import qupath.lib.gui.ExtensionClassLoader;
import qupath.lib.gui.images.stores.ImageRegionStoreFactory;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ImageServerBuilder;
import qupath.lib.images.servers.ImageServerProvider;
import qupath.lib.io.PathIO;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjectTools;
import qupath.lib.objects.classes.PathClassTools;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.scripting.QP;
import qupath.opencv.dnn.DnnTools;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;

public class TrainingApp {

    private static final Logger logger = LoggerFactory.getLogger(TrainingApp.class);

    // TensorFlow only for inference - we need PyTorch for training: https://github.com/deepjavalibrary/djl/discussions/2410
    private static final String engineName = "PyTorch";

    public static void main(String[] args) {

        if (args.length == 0) {
            logger.info("No data file specified - will demonstrate training on MNIST");
            trainMnist();
            return;
        }
        new TrainingApp().trainForData(args[0]);

    }

    private static void trainMnist() {
        int batchSize = 10_000;

        var engine = Engine.getEngine(engineName);
        engine.setRandomSeed(1243);

        Device device = getDevice(engine);

        try (NDManager manager = engine.newBaseManager(device)) {

            String name = "mlp";

            Mnist mnist = Mnist.builder()
                    .optDevice(device)
                    .optManager(manager)
                    .setSampling(batchSize, true)
                    .build();

            mnist.prepare();

            Model model = Model.newInstance(name, device);
            model.setBlock(new Mlp(28 * 28, 10, new int[] {128, 64}));

            RandomAccessDataset[] split = mnist.randomSplit(6, 4);

            long start = System.currentTimeMillis();
            trainMulticlass(model, 10, split[0], split[1]);
            long end = System.currentTimeMillis();
            logger.info("Total time: {} seconds", (end - start) / 1000.0);

        } catch (Exception e) {
            logger.error("Error during training: {}", e.getMessage(), e);
        }
    }

    private void trainForData(String pathData) {

        if (!pathData.toLowerCase().endsWith(".qpdata")) {
            logger.error("Invalid data file (should end with .qpdata): {}", pathData);
            return;
        }

        var engine = Engine.getEngine(engineName);
        engine.setRandomSeed(1243);

        Device device = getDevice(engine);

        int nOutputs = 2;

        try (Model model = buildSimpleClassifierModel(engine, "my_model", device, nOutputs)) {

            int patchSize = 48;

            ensureInitialized();

            ImageData<BufferedImage> imageData = PathIO.readImageData(new File(pathData));
            trainMulticlass(model, imageData, patchSize, 50);
        } catch (Exception e) {
            logger.error("Error during training: {}", e.getMessage(), e);
        }
    }

    private static Model buildSimpleClassifierModel(Engine engine, String name, Device device, int nOutputs) {
        var model = engine.newModel(name, device);
        model.setBlock(buildSimpleClassifierBlock(16, nOutputs));
        return model;
    }

    private static Block buildSimpleClassifierBlock(int nFilters, int nOutputs) {
        return new SequentialBlock().addAll(
                Conv2d.builder().setKernelShape(createShape(3, 3)).setFilters(nFilters).build(),
                Activation.reluBlock(),
                Dropout.builder().build(),
                Pool.maxPool2dBlock(createShape(2, 2)),
                Conv2d.builder().setKernelShape(createShape(3, 3)).setFilters(nFilters*2).build(),
                Activation.reluBlock(),
                Dropout.builder().build(),
                Pool.maxPool2dBlock(createShape(2, 2)),
                Conv2d.builder().setKernelShape(createShape(3, 3)).setFilters(nFilters).build(),
                Pool.globalMaxPool2dBlock(),
                Linear.builder().setUnits(nOutputs).build(),
                LambdaBlock.singleton(n -> n.logSoftmax(1))
        );
    }


    private static Shape createShape(int h, int w) {
        return new Shape(
                new long[]{h, w},
                new LayoutType[]{LayoutType.HEIGHT, LayoutType.WIDTH}
        );
    }

    private void ensureInitialized() {
        // Set tile cache
        long tileCacheSize = Runtime.getRuntime().maxMemory() / 4;
        var imageRegionStore = ImageRegionStoreFactory.createImageRegionStore(tileCacheSize);
        ImageServerProvider.setCache(imageRegionStore.getCache(), BufferedImage.class);

        // Set classloader to include any available extensions
        var extensionClassLoader = ExtensionClassLoader.getInstance();
        extensionClassLoader.refresh();
        ImageServerProvider.setServiceLoader(ServiceLoader.load(ImageServerBuilder.class, extensionClassLoader));
        Thread.currentThread().setContextClassLoader(extensionClassLoader);

        // Unfortunately necessary to force initialization (including GsonTools registration of some classes)
        QP.getCoreClasses();
    }


    private static Device getDevice(Engine engine) {
        if (GeneralTools.isMac())
            return Device.of("mps", 0);
        return engine.defaultDevice();
    }

    public static void trainMulticlass(Model model, ImageData<BufferedImage> imageData, int patchSize, int numEpochs) throws TranslateException, IOException {

        var trainingObjects = getTrainingObjects(imageData.getHierarchy(), imageData.getServer());

        double downsample = 4.0;

        var classifications = trainingObjects.stream().map(TrainingObject::classification).distinct().sorted().toList();
        var classificationMap = new TreeMap<String, Integer>();
        for (int i = 0; i < classifications.size(); i++) {
            classificationMap.put(classifications.get(i), i);
        }

        var manager = model.getNDManager();

        var rng = new Random(1243);
        Collections.shuffle(trainingObjects, rng);
        int indPartition = (int)Math.round(trainingObjects.size() * 0.7);
        var trainDataset = createDataset(manager, trainingObjects.subList(0, indPartition), downsample, patchSize, patchSize, classificationMap);
        var valDataset = createDataset(manager, trainingObjects.subList(indPartition, trainingObjects.size()), downsample, patchSize, patchSize, classificationMap);

        trainMulticlass(model, numEpochs, trainDataset, valDataset);

        classifyDetections(model, imageData, classifications, downsample, patchSize);
        saveImageData(imageData);
    }

    private static void classifyDetections(Model model, ImageData<BufferedImage> imageData, List<String> classifications,
                                           double downsample, int patchSize) {

        var manager = model.getNDManager();

        var toClassify = imageData.getHierarchy().getDetectionObjects()
                .stream()
                .map(detection -> new TrainingObject(null, detection, imageData.getServer()))
                .toList();

        var predictors = new HashMap<Thread, Predictor<NDList, NDList>>();
        var count = new AtomicInteger(0);

        Lists.partition(toClassify, 512).parallelStream().forEach(batchInput -> {

            try (var subManager = manager.newSubManager()) {
                var batch = new ArrayList<NDList>();
                for (var to : batchInput) {
                    var patch = to.getPatch(subManager, downsample, patchSize, patchSize);
                    batch.add(new NDList(patch));
                }
                var predictor = predictors.computeIfAbsent(
                        Thread.currentThread(),
                        t ->  model.newPredictor(new NoopTranslator()));
                var output = predictor.batchPredict(batch);

                var outputList = new NDList();
                output.stream().map(NDList::singletonOrThrow).forEach(outputList::add);
                var argMax = NDArrays.concat(outputList, 0).argMax(1);
                var inds = argMax.toLongArray();
                for (int i = 0; i < inds.length; i++) {
                    var to = batchInput.get(i);
                    to.pathObject.setClassification(classifications.get((int)inds[i]));

                    int value = count.incrementAndGet();
                    if (value % 1000 == 0) {
                        logger.info("Classified: {}/{}", value, toClassify.size());
                    }
                }
            } catch (Exception e) {
                logger.error("Error predicting patch: {}", e.getMessage(), e);
            }
        });

        predictors.values().forEach(Predictor::close);

    }

    private static boolean saveImageData(ImageData<BufferedImage> imageData) throws IOException {
        var path = imageData.getLastSavedPath();
        var file = path == null ? null : new File(path);
        if (file != null && file.exists()) {
            PathIO.writeImageData(file, imageData);
            return true;
        } else {
            return false;
        }
    }


    private static ArrayDataset createDataset(NDManager manager, Collection<? extends TrainingObject> trainingObjects,
                                              double downsample, int width, int height,
                                              Map<String, Integer> classificationMap) {

        var data = new NDList();
        var labels = new NDList();
        for (var trainingObject : trainingObjects) {
            try {
                var patch = trainingObject.getPatch(manager, downsample, width, height);
                data.add(patch);
                var label = manager.create(
                        new int[]{classificationMap.get(trainingObject.classification)}, new Shape(1));
                labels.add(label);
            } catch (IOException e) {
                logger.error("Error reading patch: {}", e.getMessage(), e);
            }
        }

        return new ArrayDataset.Builder()
                .setSampling(8, true)
                .optDevice(manager.getDevice())
                .setData(NDArrays.concat(data))
                .optLabels(NDArrays.concat(labels))
                .build();
    }

    private record TrainingObject(String classification, PathObject pathObject, ImageServer<BufferedImage> server) {

        public NDArray getPatch(NDManager manager, double downsample, int width, int height) throws IOException {
            var roi = PathObjectTools.getNucleusOrMainROI(pathObject);
            var mat = DnnTools.readPatch(server, roi, downsample, width, height);
            mat.convertTo(mat, org.bytedeco.opencv.global.opencv_core.CV_32F, 1.0 / 255.0, 0);
            return DjlTools.matToNDArray(manager, mat, "NCHW");
        }

    }

    private static List<TrainingObject> getTrainingObjects(PathObjectHierarchy hierarchy, ImageServer<BufferedImage> server) {
        List<TrainingObject> trainingObjects = new ArrayList<>();
        Set<PathObject> trainingDetections = new HashSet<>();
        for (var annotation : hierarchy.getAnnotationObjects()) {
            if (annotation.isLocked() || annotation.getPathClass() == null || PathClassTools.isIgnoredClass(annotation.getPathClass()))
                continue;

            String classification = annotation.getClassification();
            var detections = hierarchy.getAllDetectionsForROI(annotation.getROI());
            for (var d : detections) {
                if (trainingDetections.add(d))
                    trainingObjects.add(new TrainingObject(classification, d, server));
            }
        }
        return trainingObjects;
    }


    public static void trainMulticlass(Model model, int numEpochs, Dataset train, Dataset val) throws TranslateException, IOException {
        var manager = model.getNDManager();
        var device = manager.getDevice();
        // Create separately so that we can initialize the stage before the first epoch completes
        ensureToolkitInitialized();

        // Now that we have our training configuration, we should create a new trainer for our model
        Trainer trainer = createTrainer(model, Loss.softmaxCrossEntropyLoss(), device);
        initializeTrainer(trainer, train);

        EasyTrain.fit(trainer, numEpochs, train, val);
    }

    private static void initializeTrainer(Trainer trainer, Dataset dataset) throws TranslateException, IOException {
        var manager = trainer.getManager();
        try (var batch = dataset.getData(manager).iterator().next()) {
            initializeTrainer(trainer, batch.getData());
        }
    }


    private static void initializeTrainer(Trainer trainer, NDList input) {
        var shapes = input.stream().map(NDArray::getShape).toArray(Shape[]::new);
        for (int i = 0; i < shapes.length; i++) {
            var s = shapes[i];
            if (!s.isLayoutKnown()) {
                String layout = switch (s.dimension()) {
                    case 2 -> "HW";
                    case 3 -> "CHW";
                    case 4 -> "NCHW";
                    default -> throw new IllegalArgumentException("Unsupported shape: " + s);
                };
                shapes[i] = new Shape(s.getShape(), layout);
            }
        }
        trainer.initialize(shapes);
    }

    private static Trainer createTrainer(Model model, Loss loss, Device... devices) {
        var manager = model.getNDManager();
        if (devices.length == 0)
            devices = new Device[]{manager.getDevice()};

        var config = createTrainingConfig(loss, devices);
        Trainer trainer = model.newTrainer(config);
        trainer.setMetrics(new Metrics());
        return trainer;
    }

    private static TrainingConfig createTrainingConfig(Loss loss, Device... devices) {
        return new DefaultTrainingConfig(loss)
                .optDevices(devices)
                .addTrainingListeners(
                        new EpochTrainingListener(),
                        new EvaluatorTrainingListener(),
                        new DivergenceCheckTrainingListener(),
                        new LoggingTrainingListener(),
                        createJFXTrainingListenerWithStage()
                )
                .addEvaluator(new Accuracy()); // Use accuracy so we humans can understand how accurate the model is
    }


    private static JFXTrainingListener createJFXTrainingListenerWithStage() {
        var listener = new JFXTrainingListener();
        listener.getCharts().addListener(TrainingApp::handleChartsChanged);
        return listener;
    }


    /**
     * Show a stage whenever we have charts to display.
     * @param change
     */
    private static void handleChartsChanged(ListChangeListener.Change<? extends XYChart<Number, Number>> change) {
        var list = change.getList();
        if (list.isEmpty())
            return;
        var stage = new Stage();
        stage.setTitle("Training");
        var box = new VBox();
        box.getChildren().setAll(change.getList());
        box.setSpacing(5);
        stage.setScene(new Scene(new ScrollPane(box)));
        stage.show();
    }


    private static void ensureToolkitInitialized() {
        new JFXPanel();
    }

}
