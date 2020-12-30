using Microsoft.ML;
using System;
using System.Runtime.Serialization;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe base dei trainers parametrizzati
   /// </summary>
   /// <typeparam name="TModel">Tipo di parametri del modello</typeparam>
   /// <typeparam name="TTrainer">Tipo di trainer</typeparam>
   /// <typeparam name="TOptions">Tipo di opzioni</typeparam>
   public abstract class TrainerBase<TModel, TTrainer, TOptions> : IDeserializationCallback, IEstimator<ISingleFeaturePredictionTransformer<TModel>>
      where TModel : class
      where TTrainer : IEstimator<ISingleFeaturePredictionTransformer<TModel>>
      where TOptions : class, new()
   {
      #region Fields
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      private readonly MachineLearningContext _ml;
      #endregion
      #region Properties
      /// <summary>
      /// Le opzioni
      /// </summary>
      public TOptions Options { get; private set; }
      /// <summary>
      /// Il trainer
      /// </summary>
      [field: NonSerialized]
      public TTrainer Trainer { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      internal TrainerBase(MachineLearningContext ml, TOptions options = default)
      {
         _ml = ml;
         Options = options ?? new TOptions();
         Trainer = CreateTrainer(ml);
      }
      /// <summary>
      /// Funzione di creazione del trainer da implementare nelle classi derivate
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected abstract TTrainer CreateTrainer(MachineLearningContext ml);
      /// <summary>
      /// Effettua il training e ritorna un transformer
      /// </summary>
      /// <param name="input">I dati di input</param>
      /// <returns>Il transformer</returns>
      public ISingleFeaturePredictionTransformer<TModel> Fit(IDataView input) => Trainer.Fit(input);
      /// <summary>
      /// Restituisce lo schema di output dato uno schema di input
      /// </summary>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>Lo schema di output</returns>
      public SchemaShape GetOutputSchema(SchemaShape inputSchema) => Trainer.GetOutputSchema(inputSchema);
      /// <summary>
      /// Funzione di post deserializzazione
      /// </summary>
      /// <param name="sender"></param>
      void IDeserializationCallback.OnDeserialization(object sender) => Trainer = CreateTrainer(_ml);
      #endregion
   }
}
