using Microsoft.ML;
using Microsoft.ML.Data;

namespace ML.Utilities.Data
{
   /// <summary>
   /// Interfaccia per lo storage di dati
   /// </summary>
   public interface IDataStorage
   {
      #region Methods
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="extra">Sorgenti extra di dati</param>
      /// <returns>L'accesso ai dati</returns>
      IDataView LoadData(MachineLearningContext ml, params IMultiStreamSource[] extra);
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="schema">Commento contenente lo schema nei dati di tipo file testuali (ignorato negli altri)</param>
      /// <param name="extra">Sorgenti extra di dati da accodare</param>
      void SaveData(MachineLearningContext ml, IDataView data, bool schema = false, params IMultiStreamSource[] extra);
      #endregion
   }
}
