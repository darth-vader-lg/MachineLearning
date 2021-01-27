using MachineLearning.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System.Threading;

namespace MachineLearning.Model
{
   /// <summary>
   /// Classe base per i modeli ML.NET
   /// </summary>
   public abstract class ModelBaseMLNet : ModelBase<MLContext>, ITransformer
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      public ModelBaseMLNet(IContextProvider<MLContext> contextProvider = default) : base(contextProvider ?? MachineLearningContext.Default) { }
      /// <summary>
      /// Restituisce lo schema di output dato lo schema di input
      /// </summary>
      /// <param name="inputSchema">Scema di input</param>
      /// <returns>Lo schema di output</returns>
      public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => GetEvaluation(new ModelTrainerStandard()).Model?.GetOutputSchema(inputSchema);
      /// <summary>
      /// Restituisce il mapper riga a riga
      /// </summary>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>Il mappatore</returns>
      public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => GetEvaluation(new ModelTrainerStandard()).Model?.GetRowToRowMapper(inputSchema);
      /// <summary>
      /// Restituisce il modello sottoposto al training
      /// </summary>
      /// <param name="trainer">Il trainer da utilizzare</param>
      /// <param name="data">Dati di training</param>
      /// <param name="metrics">Eventuale metrica</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns></returns>
      protected sealed override ITransformer GetTrainedModel(IModelTrainer trainer, IDataAccess data, out object metrics, CancellationToken cancellation)
      {
         Channel.CheckValue(trainer, nameof(trainer));
         Channel.CheckValue(data, nameof(data));
         return trainer.GetTrainedModel(this, data, out metrics, cancellation);
      }
      /// <summary>
      /// Carica i dati da uno storage
      /// </summary>
      /// <param name="dataStorage">Storage di dati</param>
      /// <returns>La vista di dati</returns>
      public sealed override IDataAccess LoadData(IDataStorage dataStorage)
      {
         Channel.CheckValue(dataStorage, nameof(dataStorage));
         var options = (this as ITextLoaderOptions)?.TextLoaderOptions ?? new TextLoader.Options() { Columns = (this as IInputSchema)?.InputSchema?.ToTextLoaderColumns() };
         return dataStorage.LoadData(Context, options);
      }
      public sealed override ITransformer LoadModel(IModelStorage modelStorage, out DataViewSchema schema)
      {
         Channel.CheckValue(modelStorage, nameof(modelStorage));
         return modelStorage.LoadModel(Context, out schema);
      }
      /// <summary>
      /// Effettua il salvataggio del modello
      /// </summary>
      /// <param name="ctx">Contesto di salvataggio</param>
      void ICanSaveModel.Save(ModelSaveContext ctx)
      {
         var evaluation = GetEvaluation(new ModelTrainerStandard());
         if (evaluation.Model != null && evaluation.ModelStorage != null)
            evaluation.ModelStorage.SaveModel(Context, evaluation.Model, evaluation.InputSchema);
      }
      /// <summary>
      /// Salva i dati in uno storage
      /// </summary>
      /// <param name="dataStorage">Storage di dati</param>
      /// <param name="data">Dati</param>
      public sealed override void SaveData(IDataStorage dataStorage, IDataAccess data)
      {
         Channel.CheckValue(dataStorage, nameof(dataStorage));
         Channel.CheckValue(data, nameof(data));
         var options = (this as ITextLoaderOptions)?.TextLoaderOptions ?? new TextLoader.Options() { Columns = (this as IInputSchema)?.InputSchema?.ToTextLoaderColumns() };
         dataStorage.SaveData(Context, data, options);
      }
      /// <summary>
      /// Salva il modello in uno storage
      /// </summary>
      /// <param name="modelStorage">Storage del modello</param>
      /// <param name="model">Modello</param>
      /// <param name="schema">Lo schema del modello</param>
      /// <returns>Il modello</returns>
      public sealed override void SaveModel(IModelStorage modelStorage, ITransformer model, DataViewSchema schema)
      {
         Channel.CheckValue(modelStorage, nameof(modelStorage));
         Channel.CheckValue(model, nameof(model));
         modelStorage.SaveModel(Context, model, schema);
      }
      /// <summary>
      /// Trasforma i dati di input per il modello
      /// </summary>
      /// <param name="input">Vista di dati di input</param>
      /// <returns>I dati trasformati</returns>
      public IDataView Transform(IDataView input) => GetEvaluation(new ModelTrainerStandard()).Model?.Transform(input);
      #endregion
   }
}
