package qupath.ext.djl.training;

import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.listener.EvaluatorTrainingListener;
import ai.djl.training.listener.TrainingListener;
import javafx.application.Platform;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.ReadOnlyIntegerProperty;
import javafx.beans.property.ReadOnlyObjectProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import qupath.fx.utils.FXUtils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Objects;

/**
 * A {@link TrainingListener} that updates JavaFX charts with training metrics.
 * Note that properties wil be updated in the JavaFX application thread.
 */
public class JFXTrainingListener implements TrainingListener {

    private static final String EVALUATOR_KEY = JFXTrainingListener.class.getName() + ".evaluator-name";

    public enum Status {
        NOT_STARTED, TRAINING, COMPLETE
    }

    private record MetricData(XYChart<Number, Number> chart, String evaluatorName, double metricTrain, double metricVal) {}

    private boolean initialized = false;

    private final ObservableList<LineChart<Number, Number>> charts = FXCollections.observableArrayList();
    private final ObservableList<LineChart<Number, Number>> chartsUnmodifiable = FXCollections.unmodifiableObservableList(charts);

    private final IntegerProperty epochProperty = new SimpleIntegerProperty();
    private final ObjectProperty<Status> statusProperty = new SimpleObjectProperty<>();

    /**
     * Create a new training listener that starts at epoch 0.
     */
    public JFXTrainingListener() {
        this(0);
    }

    /**
     * Create a new training listener that starts at the given epoch.
     * @param startEpoch
     */
    public JFXTrainingListener(int startEpoch) {
        epochProperty.set(startEpoch);
    }

    /**
     * Returns the property that tracks the current epoch.
     * @return the epoch property
     */
    public ReadOnlyIntegerProperty epochProperty() {
        return epochProperty;
    }

    /**
     * Returns the current epoch.
     * @return the current epoch
     */
    public int getEpoch() {
        return epochProperty.get();
    }

    /**
     * Returns the list of charts that are updated with training metrics.
     * @return the list of charts
     */
    public ObservableList<LineChart<Number, Number>> getCharts() {
        return chartsUnmodifiable;
    }

    /**
     * Returns the property that tracks the current status.
     * @return
     */
    public ReadOnlyObjectProperty<Status> statusProperty() {
        return statusProperty;
    }

    /**
     * Returns the current status.
     * @return
     */
    public Status getStatus() {
        return statusProperty.get();
    }

    /**
     * Set the evaluators to display on the charts.
     * <p>
     * Either this or {@link #setEvaluators(String...)}should be called exactly once before training begins,
     * or else charts will be generated for all evaluators as soon as the trainer is created.
     * @param evaluators
     * @see #setEvaluators(String...)
     */
    public void setEvaluators(Collection<? extends Evaluator> evaluators) {
        setEvaluators(evaluators.stream().map(Evaluator::getName).toArray(String[]::new));
    }

    /**
     * Set the evaluators to display on the charts by name.
     * <p>
     * Either this or {@link #setEvaluators(Collection)}should be called exactly once before training begins,
     * or else charts will be generated for all evaluators as soon as the trainer is created.
     * @param evaluatorNames
     * @see #setEvaluators(Collection)
     */
    public void setEvaluators(String... evaluatorNames) {
        if (!Platform.isFxApplicationThread()) {
            FXUtils.runOnApplicationThread(() -> setEvaluators(evaluatorNames));
            return;
        }
        if (initialized) {
            throw new RuntimeException("Charts have already been initialized!");
        }
        var chartsToAdd = new ArrayList<LineChart<Number, Number>>();
        for (var name : evaluatorNames) {
            var chart = new LineChart<>(new NumberAxis(), new NumberAxis());
            chart.setTitle(name);
            chart.getXAxis().setLabel("Epoch");
            chart.getYAxis().setLabel("Value");
            chart.setMaxSize(Double.MAX_VALUE, Double.MAX_VALUE);

            var seriesTrain = new XYChart.Series<Number, Number>();
            seriesTrain.setName("Train");
            chart.getData().add(seriesTrain);

            var seriesVal = new XYChart.Series<Number, Number>();
            seriesVal.setName("Validation");
            chart.getData().add(seriesVal);

            chart.getProperties().put(EVALUATOR_KEY, name);
            chartsToAdd.add(chart);
        }
        charts.setAll(chartsToAdd);
        initialized = true;
    }

    @Override
    public void onEpoch(Trainer trainer) {
        var evaluators = trainer.getEvaluators();
        var metrics = trainer.getMetrics();
        var metricData = new ArrayList<MetricData>();
        for (var chart : charts) {
            var name = chart.getProperties().get(EVALUATOR_KEY);
            var evaluator = evaluators.stream()
                    .filter(e -> Objects.equals(name, e.getName()))
                    .findFirst()
                    .orElse(null);
            if (evaluator == null) {
                continue;
            }
            var metricTrain = metrics.latestMetric(
                    EvaluatorTrainingListener.metricName(evaluator, EvaluatorTrainingListener.TRAIN_EPOCH)
            );
            var metricVal = metrics.latestMetric(
                    EvaluatorTrainingListener.metricName(evaluator, EvaluatorTrainingListener.VALIDATE_EPOCH)
            );
            metricData.add(new MetricData(chart, evaluator.getName(), metricTrain.getValue(), metricVal.getValue()));
        }
        FXUtils.runOnApplicationThread(() -> {
            epochProperty.set(epochProperty.get() + 1);
            appendMetrics(metricData);
        });
    }

    private void appendMetrics(Collection<MetricData> metricData) {
        for (var metric : metricData) {
            var chart = metric.chart;
            var seriesTrain = chart.getData().get(0).getData();
            seriesTrain.add(new XYChart.Data<>(seriesTrain.size(), metric.metricTrain));
            var seriesVal = chart.getData().get(1).getData();
            seriesVal.add(new XYChart.Data<>(seriesVal.size(), metric.metricVal));
        }
    }

    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {
    }

    @Override
    public void onValidationBatch(Trainer trainer, BatchData batchData) {
    }

    @Override
    public void onTrainingBegin(Trainer trainer) {
        FXUtils.runOnApplicationThread(() -> {
            if (!initialized) {
                setEvaluators(trainer.getEvaluators());
            }
            statusProperty.set(Status.TRAINING);
        });
    }

    @Override
    public void onTrainingEnd(Trainer trainer) {
        FXUtils.runOnApplicationThread(() -> statusProperty.set(Status.COMPLETE));
    }

}
