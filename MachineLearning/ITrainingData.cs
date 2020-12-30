using Microsoft.ML;
using Microsoft.ML.Data;

namespace MachineLearning
{
   /// <summary>
   /// Interfaccia per lo storage di dati di training
   /// </summary>
   public interface ITrainingData
   {
      #region Methods
      /// <summary>
      /// Carica i dati di training
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="opt">Opzioni di testo</param>
      /// <param name="extra">Sorgenti extra di dati</param>
      /// <returns>L'accesso ai dati</returns>
      IDataView LoadTrainingData(MachineLearningContext ml, TextLoader.Options opt = default, params IMultiStreamSource[] extra);
      /// <summary>
      /// Salva i dati di training
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="opt">Opzioni di testo</param>
      /// <param name="schema">Commento contenente lo schema nei dati di tipo file testuali (ignorato negli altri)</param>
      /// <param name="extra">Sorgenti extra di dati da accodare</param>
      void SaveTrainingData(MachineLearningContext ml, IDataView data, TextLoader.Options opt = default, bool schema = false, params IMultiStreamSource[] extra);
      #endregion
   }
}
