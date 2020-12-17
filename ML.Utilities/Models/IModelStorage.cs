using Microsoft.ML;

namespace ML.Utilities.Models
{
   /// <summary>
   /// Interfaccia per i gestori di storage dei modelli
   /// </summary>
   public interface IModelStorage
   {
      #region Methods
      /// <summary>
      /// Funzione di caricamento modello
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="schema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      ITransformer LoadModel(MachineLearningContext ml, out DataViewSchema schema);
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="model">Modello da salvare</param>
      /// <param name="schema">Schema di input del modello</param>
      void SaveModel(MachineLearningContext ml, ITransformer model, DataViewSchema schema);
      #endregion
   }
}